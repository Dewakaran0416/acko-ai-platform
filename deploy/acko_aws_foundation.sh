#!/usr/bin/env bash
# ==============================================================================
# acko_aws_foundation.sh  -  Acko AI Platform, Step 15
#
# Builds the AWS pieces the platform needs, all in Sydney (ap-southeast-2):
#   1. two firewall groups   acko-ec2-sg (web port 8501) and acko-rds-sg (port 5432)
#   2. the PostgreSQL database   acko-db   (private: only the server can reach it)
#   3. a permission role   acko-ec2-role   (use the S3 bucket + Session Manager)
#   4. the server   acko-server   (Amazon Linux 2023, t3.small) with a fixed address
#
# Run it in AWS CloudShell:        bash acko_aws_foundation.sh
# Look at the state again later:   bash acko_aws_foundation.sh status
#
# Safe to run again: whatever already exists is reused, never duplicated.
# The database password is typed (hidden) and is never printed or saved.
# ==============================================================================
set -uo pipefail
MODE="${1:-run}"

export AWS_PAGER=""
R="ap-southeast-2"
export AWS_DEFAULT_REGION="$R"
export AWS_REGION="$R"

BUCKET="${ACKO_BUCKET:-acko-claims-dewa-2026}"
DB_ID="acko-db"
DB_NAME="acko"
DB_USER="acko_admin"
SG_EC2="acko-ec2-sg"
SG_RDS="acko-rds-sg"
ROLE="acko-ec2-role"
PROFILE="acko-ec2-profile"
SERVER="acko-server"
EIP_NAME="acko-eip"
TAG_PROJECT="{Key=Project,Value=acko-ai-platform}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '   [ ok ] %s\n' "$*"; }
note() { printf '          %s\n' "$*"; }
warn() { printf '   [warn] %s\n' "$*"; }
die()  { printf '\n   [STOP] %s\n' "$*"; printf '   Take a screenshot of this screen and send it to Claude.\n\n'; exit 1; }
real() { [ -n "${1:-}" ] && [ "$1" != "None" ]; }

# ------------------------------------------------------------------ 1 who and where
stage_identity() {
  say "1 of 6  Who is signed in, and where"
  local ident
  ident="$(aws sts get-caller-identity --query '[Account,Arn]' --output text 2>&1)" \
    || die "AWS did not accept this session: $ident"
  ACCOUNT="$(printf '%s' "$ident" | awk '{print $1}')"
  WHO="$(printf '%s' "$ident" | awk '{print $2}')"
  ok "account $ACCOUNT"
  ok "signed in as $WHO"
  ok "region $R (Sydney) - every command below is sent to this region"
}

stage_plan() {
  local plan p_type p_credit p_end ans
  plan="$(aws freetier get-account-plan-state --region us-east-1 \
          --query '[accountPlanType,accountPlanStatus,accountPlanRemainingCredits.amount,accountPlanExpirationDate]' \
          --output text 2>/dev/null)" || plan=""
  if [ -z "$plan" ]; then
    note "(The Free plan state could not be read. That is fine - check Billing > Credits yourself.)"
    return 0
  fi
  read -r p_type _ p_credit p_end <<< "$plan"
  p_type="${p_type:-?}"; p_credit="${p_credit:-?}"; p_end="${p_end:-?}"
  if [ "$p_type" = "FREE" ]; then
    ok "AWS Free plan is active: credits left about $p_credit USD, the plan ends ${p_end%%T*}"
  else
    warn "The AWS Free plan is not active on this account (plan type: $p_type)."
    note "Unless you have other AWS credits, the server and database below cost real money:"
    note "roughly 1 to 2 US dollars per day while they exist."
    printf '          Type  yes  to go on, anything else to stop: '
    read -r ans
    if [ "$ans" != "yes" ]; then
      printf '\n   Stopped on purpose. Nothing was created.\n\n'
      exit 0
    fi
  fi
}

# ------------------------------------------------------------------ 2 network
stage_network() {
  say "2 of 6  The network"
  VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text 2>&1)"
  case "$VPC" in
    vpc-*) ok "default network $VPC" ;;
    *) die "No default network was found in Sydney. AWS said: $VPC" ;;
  esac
}

# ------------------------------------------------------------------ 3 firewall groups
sg_id() {
  aws ec2 describe-security-groups --filters "Name=group-name,Values=$1" "Name=vpc-id,Values=$VPC" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null
}

sg_make() {   # $1 = name, $2 = description; prints the group id
  local id
  id="$(sg_id "$1")"
  if ! real "$id"; then
    id="$(aws ec2 create-security-group --group-name "$1" --description "$2" --vpc-id "$VPC" \
          --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$1},$TAG_PROJECT]" \
          --query GroupId --output text 2>&1)" || { printf '%s\n' "$id" >&2; return 1; }
  fi
  printf '%s\n' "$id"
}

allow() {     # $1 = what the rule does, then the permission text
  local out
  out="$(aws ec2 authorize-security-group-ingress --group-id "$2" --ip-permissions "$3" 2>&1)" \
    && { ok "$1"; return 0; }
  case "$out" in
    *InvalidPermission.Duplicate*) ok "$1 (already there)"; return 0 ;;
  esac
  printf '%s\n' "$out" >&2
  return 1
}

stage_firewall() {
  say "3 of 6  Firewall groups"
  EC2_SG="$(sg_make "$SG_EC2" "Acko app server - web port 8501")" || die "The firewall group $SG_EC2 could not be created."
  RDS_SG="$(sg_make "$SG_RDS" "Acko database - port 5432 from the server only")" || die "The firewall group $SG_RDS could not be created."
  ok "$SG_EC2 = $EC2_SG"
  ok "$SG_RDS = $RDS_SG"
  allow "web port 8501 is open to visitors" "$EC2_SG" \
        "IpProtocol=tcp,FromPort=8501,ToPort=8501,IpRanges=[{CidrIp=0.0.0.0/0,Description=Streamlit-app}]" \
    || die "The web rule could not be added."
  allow "database port 5432 is open only to $SG_EC2" "$RDS_SG" \
        "IpProtocol=tcp,FromPort=5432,ToPort=5432,UserIdGroupPairs=[{GroupId=$EC2_SG,Description=Acko-server}]" \
    || die "The database rule could not be added."
}

# ------------------------------------------------------------------ 4 database
ask_password() {
  local p1 p2
  printf '\n   Choose a password for the database (12 or more letters and digits, no spaces or symbols).\n'
  printf '   Typing is hidden. Write it down somewhere safe - you need it again in the next steps.\n'
  while true; do
    printf '   Password: '; read -r -s p1; printf '\n'
    printf '   Again   : '; read -r -s p2; printf '\n'
    if [ "$p1" != "$p2" ]; then
      printf '   The two entries are different. Try again.\n'; continue
    fi
    if ! [[ "$p1" =~ ^[A-Za-z0-9]{12,64}$ ]]; then
      printf '   Use only letters and digits, 12 to 64 of them. Try again.\n'; continue
    fi
    DB_PASS="$p1"
    return 0
  done
}

stage_database() {
  say "4 of 6  The database (PostgreSQL on Amazon RDS)"
  local status subnet_group subnets out
  local -a subnet_list
  status="$(aws rds describe-db-instances --db-instance-identifier "$DB_ID" \
            --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null)" || status=""
  if real "$status"; then
    ok "database $DB_ID already exists (status: $status) - left as it is"
    return 0
  fi

  subnet_group="$(aws rds describe-db-subnet-groups \
                  --query "DBSubnetGroups[?VpcId=='$VPC'].DBSubnetGroupName | [0]" --output text 2>/dev/null)" || subnet_group=""
  if ! real "$subnet_group"; then
    subnets="$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" --query 'Subnets[].SubnetId' --output text 2>&1)" \
      || die "The subnets of the default network could not be listed: $subnets"
    read -r -a subnet_list <<< "$subnets"
    out="$(aws rds create-db-subnet-group --db-subnet-group-name acko-db-subnets \
           --db-subnet-group-description "Acko database subnets" --subnet-ids "${subnet_list[@]}" 2>&1)" \
      || die "The database subnet group could not be created: $out"
    subnet_group="acko-db-subnets"
  fi
  ok "database subnet group $subnet_group"

  ask_password
  printf '   Creating the database ... '
  out="$(aws rds create-db-instance \
         --db-instance-identifier "$DB_ID" \
         --engine postgres \
         --db-instance-class db.t4g.micro \
         --allocated-storage 20 --storage-type gp3 --storage-encrypted \
         --db-name "$DB_NAME" \
         --master-username "$DB_USER" --master-user-password "$DB_PASS" \
         --vpc-security-group-ids "$RDS_SG" \
         --db-subnet-group-name "$subnet_group" \
         --no-publicly-accessible --no-multi-az \
         --backup-retention-period 1 \
         --no-deletion-protection \
         --tags "Key=Name,Value=$DB_ID" "Key=Project,Value=acko-ai-platform" \
         --query 'DBInstance.DBInstanceStatus' --output text 2>&1)" \
    || { unset DB_PASS; printf '\n'; die "The database could not be created: $out"; }
  unset DB_PASS
  printf 'started (status: %s)\n' "$out"
  ok "the database is private (not reachable from the internet) and encrypted"
  note "It needs about 10 minutes to become ready. The script does not wait for it."
}

# ------------------------------------------------------------------ 5 permission role
stage_role() {
  say "5 of 6  The server's permission role"
  local out has
  if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    ok "role $ROLE already exists"
  else
    cat > "$TMP/trust.json" <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
    out="$(aws iam create-role --role-name "$ROLE" \
           --assume-role-policy-document "file://$TMP/trust.json" \
           --description "Acko server: S3 bucket access and Session Manager" \
           --tags "Key=Project,Value=acko-ai-platform" 2>&1)" \
      || die "The role could not be created: $out"
    ok "role $ROLE created"
  fi

  cat > "$TMP/access.json" <<EOF
{"Version":"2012-10-17","Statement":[
 {"Sid":"BucketLevel","Effect":"Allow","Action":["s3:ListBucket","s3:GetBucketLocation"],"Resource":"arn:aws:s3:::$BUCKET"},
 {"Sid":"ObjectLevel","Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
 {"Sid":"FindTheDatabase","Effect":"Allow","Action":"rds:DescribeDBInstances","Resource":"*"}
]}
EOF
  out="$(aws iam put-role-policy --role-name "$ROLE" --policy-name acko-app-access \
         --policy-document "file://$TMP/access.json" 2>&1)" || die "The access policy could not be set: $out"
  out="$(aws iam attach-role-policy --role-name "$ROLE" \
         --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>&1)" \
    || die "Session Manager access could not be added: $out"
  ok "the role may use ONLY the bucket $BUCKET, read the database address, and use Session Manager"

  if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
    out="$(aws iam create-instance-profile --instance-profile-name "$PROFILE" 2>&1)" \
      || die "The instance profile could not be created: $out"
  fi
  has="$(aws iam get-instance-profile --instance-profile-name "$PROFILE" \
         --query "InstanceProfile.Roles[?RoleName=='$ROLE'].RoleName | [0]" --output text 2>/dev/null)" || has=""
  if [ "$has" != "$ROLE" ]; then
    out="$(aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE" 2>&1)" \
      || die "The role could not be put into the instance profile: $out"
  fi
  ok "instance profile $PROFILE holds the role"
}

# ------------------------------------------------------------------ 6 server
find_server() {
  aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$SERVER" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId | [0]' --output text 2>/dev/null
}

stage_server() {
  say "6 of 6  The server (Amazon EC2) and its fixed address"
  local iid ami root try out alloc st
  iid="$(find_server)"
  if real "$iid"; then
    ok "server $SERVER already exists ($iid)"
    st="$(aws ec2 describe-instances --instance-ids "$iid" \
          --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)" || st=""
    if [ "$st" = "stopping" ]; then
      aws ec2 wait instance-stopped --instance-ids "$iid" >/dev/null 2>&1
      st="stopped"
    fi
    if [ "$st" = "stopped" ]; then
      out="$(aws ec2 start-instances --instance-ids "$iid" 2>&1)" || die "The stopped server could not be started: $out"
      ok "the server was stopped - starting it again"
    fi
  else
    ami="$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
           --query Parameter.Value --output text 2>&1)" || die "The newest Amazon Linux image could not be found: $ami"
    root="$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text 2>/dev/null)" || root=""
    real "$root" || root="/dev/xvda"
    ok "image $ami (Amazon Linux 2023), disk $root"

    cat > "$TMP/userdata.sh" <<'EOF'
#!/bin/bash
# Runs once, as root, when the server starts for the first time.
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
# 1 GB of swap as a safety net while Docker builds the image
dd if=/dev/zero of=/swapfile bs=1M count=1024
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
touch /var/lib/acko-setup-done
EOF

    iid=""
    for try in 1 2 3 4 5 6 7 8; do
      iid="$(aws ec2 run-instances \
             --image-id "$ami" --instance-type t3.small \
             --security-group-ids "$EC2_SG" \
             --iam-instance-profile "Name=$PROFILE" \
             --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
             --block-device-mappings "DeviceName=$root,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}" \
             --user-data "file://$TMP/userdata.sh" \
             --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$SERVER},$TAG_PROJECT]" \
                                  "ResourceType=volume,Tags=[{Key=Name,Value=$SERVER},$TAG_PROJECT]" \
             --query 'Instances[0].InstanceId' --output text 2>"$TMP/run.err")" && break
      if grep -qiE 'instance ?profile' "$TMP/run.err"; then
        note "the new role is still spreading inside AWS - waiting 10 seconds (try $try of 8) ..."
        sleep 10
      else
        break
      fi
    done
    case "$iid" in
      i-*) ok "server $iid is starting" ;;
      *) die "The server could not be started: $(cat "$TMP/run.err")" ;;
    esac
  fi

  printf '   Waiting for the server to run ... '
  out="$(aws ec2 wait instance-running --instance-ids "$iid" 2>&1)" || { printf '\n'; die "The server did not reach the running state: $out"; }
  printf 'running\n'

  alloc="$(aws ec2 describe-addresses --filters "Name=tag:Name,Values=$EIP_NAME" \
           --query 'Addresses[0].AllocationId' --output text 2>/dev/null)" || alloc=""
  if ! real "$alloc"; then
    alloc="$(aws ec2 allocate-address --domain vpc \
             --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$EIP_NAME},$TAG_PROJECT]" \
             --query AllocationId --output text 2>&1)" || die "A fixed address could not be reserved: $alloc"
  fi
  out="$(aws ec2 associate-address --instance-id "$iid" --allocation-id "$alloc" 2>&1)" \
    || die "The fixed address could not be attached to the server: $out"
  ok "fixed address attached to the server"
}

# ------------------------------------------------------------------ 7 summary
summary() {
  say "Summary"
  local ec2sg rdssg dbline db_s db_e db_v db_p iid state itype pubip ssm
  ec2sg="$(sg_id "$SG_EC2")"; rdssg="$(sg_id "$SG_RDS")"
  dbline="$(aws rds describe-db-instances --db-instance-identifier "$DB_ID" \
            --query 'DBInstances[0].[DBInstanceStatus,Endpoint.Address,EngineVersion,PubliclyAccessible]' --output text 2>/dev/null)" || dbline=""
  read -r db_s db_e db_v db_p <<< "$dbline"
  iid="$(find_server)"
  state=""; itype=""; pubip=""; ssm=""
  if real "$iid"; then
    state="$(aws ec2 describe-instances --instance-ids "$iid" --query 'Reservations[0].Instances[0].[State.Name,InstanceType]' --output text 2>/dev/null)" || state=""
    read -r state itype <<< "$state"
    pubip="$(aws ec2 describe-addresses --filters "Name=tag:Name,Values=$EIP_NAME" --query 'Addresses[0].PublicIp' --output text 2>/dev/null)" || pubip=""
    ssm="$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$iid" \
           --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)" || ssm=""
  fi

  printf '\n'
  printf '   Region .......... %s (Sydney)\n' "$R"
  printf '   Account ......... %s\n' "${ACCOUNT:-?}"
  printf '   Firewall groups . %s = %s\n' "$SG_EC2" "$(real "$ec2sg" && echo "$ec2sg" || echo "MISSING")"
  printf '                     %s = %s\n' "$SG_RDS" "$(real "$rdssg" && echo "$rdssg" || echo "MISSING")"
  if real "${db_s:-}"; then
    printf '   Database ........ %s   status: %s   (PostgreSQL %s)\n' "$DB_ID" "$db_s" "${db_v:-?}"
    if real "${db_e:-}"; then
      printf '   Database address  %s\n' "$db_e"
    else
      printf '   Database address  not ready yet (it appears when the status becomes "available")\n'
    fi
    if [ "${db_p:-}" = "False" ] || [ "${db_p:-}" = "false" ]; then
      printf '   Database access . private - not reachable from the internet\n'
    else
      printf '   Database access . WARNING: the database is public (publicly accessible: %s)\n' "${db_p:-?}"
    fi
  else
    printf '   Database ........ MISSING\n'
  fi
  if real "$iid"; then
    printf '   Server .......... %s   %s   %s   state: %s\n' "$SERVER" "$iid" "${itype:-?}" "${state:-?}"
    printf '   Fixed address ... %s\n' "$(real "$pubip" && echo "$pubip" || echo "MISSING")"
    printf '   The app will be at  http://%s:8501\n' "${pubip:-?}"
    if [ "$ssm" = "Online" ]; then
      printf '   Session Manager . online - you can open a terminal on the server\n'
    else
      printf '   Session Manager . not online yet (normal in the first 2-3 minutes after the server starts)\n'
    fi
  else
    printf '   Server .......... MISSING\n'
  fi
  printf '\n   To look at this again later:  bash acko_aws_foundation.sh status\n\n'
}

# ------------------------------------------------------------------ main
if [ "$MODE" = "status" ]; then
  stage_identity
  stage_network
  summary
  exit 0
fi

stage_identity
stage_plan
stage_network
stage_firewall
stage_database
stage_role
stage_server
summary
