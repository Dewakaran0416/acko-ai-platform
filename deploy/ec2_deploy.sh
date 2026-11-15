#!/usr/bin/env bash
# ==============================================================================
# ec2_deploy.sh  -  Acko AI Platform, Step 17   (run this ON the EC2 server)
#
#   sudo bash /opt/acko/src/deploy/ec2_deploy.sh                 first deployment, or after "git pull"
#   sudo bash /opt/acko/src/deploy/ec2_deploy.sh --new-secrets   type the three secrets again
#
# What it does:
#   1. checks that the server is ready (Docker running, first start finished)
#   2. finds the address of the database acko-db by itself
#   3. asks for the three secrets (typing is hidden) and saves them in /opt/acko/acko.env
#      (only the root user can read that file; it is never put into Git or into the Docker image)
#   4. builds the Docker image (first time: 8 to 12 minutes)
#   5. starts the container "acko" on port 8501 (it also starts again after a reboot)
#   6. runs the deployment check inside the container and shows the public address
#
# Safe to run again: the saved secrets are reused, and finished build steps are kept.
# ==============================================================================
set -uo pipefail
export AWS_PAGER=""

NEW_SECRETS=0
case "${1:-}" in
  "") ;;
  --new-secrets) NEW_SECRETS=1 ;;
  *) printf 'Usage:  sudo bash %s [--new-secrets]\n' "$0"; exit 2 ;;
esac

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${ACKO_HOME:-/opt/acko}"
ENV_FILE="$HOME_DIR/acko.env"
MARKER="${ACKO_MARKER:-/var/lib/acko-setup-done}"
BUILD_LOG="${ACKO_BUILD_LOG:-/var/log/acko-build.log}"
BEAT="${ACKO_BEAT:-20}"
TRIES="${ACKO_WAIT_TRIES:-60}"      # the app gets 60 x 3 seconds to answer
R="${ACKO_REGION:-ap-southeast-2}"
export AWS_DEFAULT_REGION="$R"
BUCKET="${ACKO_BUCKET:-acko-claims-dewa-2026}"
DB_ID="acko-db"
DB_NAME="acko"
DB_USER="acko_admin"
IMAGE="acko:latest"
NAME="acko"
PORT=8501

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '   [ ok ] %s\n' "$*"; }
note() { printf '          %s\n' "$*"; }
warn() { printf '   [warn] %s\n' "$*"; }
die()  { printf '\n   [STOP] %s\n' "$*"; printf '   Take a screenshot of this screen and send it to Claude.\n\n'; exit 1; }
later() { printf '\n   [WAIT] %s\n   Nothing is wrong. Just run the same command again when the time has passed.\n\n' "$*"; exit 1; }

# ------------------------------------------------------------------ 1 the server
stage_server() {
  say "1 of 6  Is the server ready?"
  local free commit
  [ "$(id -u)" -eq 0 ] || die "Run this with sudo:   sudo bash $0"
  command -v docker >/dev/null 2>&1 || die "Docker is not installed yet. The server may still be setting itself up: wait 3 minutes and run this again."
  [ -e "$MARKER" ] || later "The server has not finished its first start yet (Docker and the swap file are still being set up). Wait 2 to 3 minutes."
  docker info >/dev/null 2>&1 || die "Docker is installed but does not answer. Try:  sudo systemctl start docker   and run this again."
  [ -f "$SRC/Dockerfile" ] && [ -f "$SRC/requirements.txt" ] && [ -f "$SRC/constraints.txt" ] && [ -f "$SRC/app.py" ] \
    || die "$SRC does not look like the Acko project folder (Dockerfile, requirements.txt, constraints.txt or app.py is missing)."
  ok "Docker is running, the server finished its first start"

  free="$(df -Pk "$SRC" | awk 'NR==2 {printf "%d", $4 / 1048576}')"
  if [ "${free:-0}" -lt 8 ]; then
    die "Only $free GB of disk are free and the image needs about 8 GB while it is built. Free some space:  sudo docker system prune -af"
  fi
  ok "$free GB of disk are free"

  # Files written on Windows may carry Windows line endings; Linux wants plain ones (harmless if already plain).
  local f
  for f in Dockerfile requirements.txt constraints.txt .dockerignore; do
    if [ -f "$SRC/$f" ] && grep -q $'\r' "$SRC/$f"; then sed -i 's/\r$//' "$SRC/$f"; note "(removed Windows line endings from $f)"; fi
  done

  commit="$(git -C "$SRC" log -1 --format='%h  %s' 2>/dev/null)" || commit=""
  if [ -n "$commit" ]; then ok "project version  $commit"; fi
}

# ------------------------------------------------------------------ 2 the database
stage_database() {
  say "2 of 6  The database"
  local line status endpoint
  line="$(aws rds describe-db-instances --region "$R" --db-instance-identifier "$DB_ID" \
          --query 'DBInstances[0].[DBInstanceStatus,Endpoint.Address]' --output text 2>&1)" \
    || die "The database $DB_ID could not be looked up: $line"
  read -r status endpoint <<< "$line"
  if [ "$status" != "available" ]; then
    later "The database is not ready yet (status: ${status:-unknown}). It needs about 10 minutes after Step 15. Wait a few minutes."
  fi
  case "$endpoint" in
    ""|None) die "The database is available but has no address. Look at it in the RDS page of the AWS console." ;;
  esac
  DB_HOST="$endpoint"
  ok "database $DB_ID is available at $DB_HOST"
  if timeout 6 bash -c "exec 3<>/dev/tcp/$DB_HOST/5432" 2>/dev/null; then
    ok "this server can reach the database port 5432"
  else
    warn "this server cannot reach port 5432 of the database. The deployment check at the end will say more."
    note "Look at the firewall group acko-rds-sg: it must allow port 5432 from acko-ec2-sg."
  fi
}

# ------------------------------------------------------------------ 3 the secrets
saved() {   # the saved value of one setting (nothing if there is none)
  if [ -f "$ENV_FILE" ]; then sed -n "s/^$1=//p" "$ENV_FILE" | head -n 1; fi
  return 0
}

ask_secret() {   # $1 label, $2 allowed pattern, $3 hint, $4 = twice (optional); the answer is left in $SECRET
  local label="$1" pattern="$2" hint="$3" twice="${4:-}" a b
  while true; do
    printf '   %s: ' "$label"
    read -r -s a || { printf '\n'; die "No input was received."; }
    printf '\n'
    if [ -n "$twice" ]; then
      printf '   Once more: '
      read -r -s b || { printf '\n'; die "No input was received."; }
      printf '\n'
      if [ "$a" != "$b" ]; then printf '   The two entries are different. Try again.\n'; continue; fi
    fi
    if [[ "$a" =~ $pattern ]]; then SECRET="$a"; return 0; fi
    printf '   %s Try again.\n' "$hint"
  done
}

write_env() {
  local tmp
  mkdir -p "$HOME_DIR"
  tmp="$(mktemp "$HOME_DIR/.acko.env.XXXXXX")" || die "The settings file could not be created in $HOME_DIR."
  {
    printf 'AWS_REGION=%s\n' "$R"
    printf 'S3_BUCKET=%s\n' "$BUCKET"
    printf 'MODELS_FROM_S3=true\n'
    printf 'DB_HOST=%s\n' "$DB_HOST"
    printf 'DB_PORT=5432\n'
    printf 'DB_NAME=%s\n' "$DB_NAME"
    printf 'DB_USER=%s\n' "$DB_USER"
    printf 'DB_PASSWORD=%s\n' "$DB_PASS"
    printf 'DB_SSLMODE=require\n'
    printf 'GEMINI_API_KEY=%s\n' "$GEMINI"
    printf 'MANAGER_PASSWORD=%s\n' "$MANAGER"
    if [ -n "$GEMINI_MODEL" ]; then printf 'GEMINI_MODEL=%s\n' "$GEMINI_MODEL"; fi
  } > "$tmp"
  chmod 600 "$tmp"
  mv -f "$tmp" "$ENV_FILE"
}

stage_secrets() {
  say "3 of 6  Settings and secrets"
  DB_PASS=""; GEMINI=""; MANAGER=""; SECRET=""
  GEMINI_MODEL="$(saved GEMINI_MODEL)"
  if [ "$NEW_SECRETS" -eq 0 ]; then
    DB_PASS="$(saved DB_PASSWORD)"; GEMINI="$(saved GEMINI_API_KEY)"; MANAGER="$(saved MANAGER_PASSWORD)"
  fi
  if [ -n "$DB_PASS" ] && [ -n "$GEMINI" ] && [ -n "$MANAGER" ]; then
    ok "the three secrets are already saved (to type them again, run this script with --new-secrets)"
  else
    note "Typing is hidden. Nothing you type here is shown, printed or sent anywhere else."
    if [ -z "$DB_PASS" ]; then
      note "1) Database password: the one you chose in Step 15."
      ask_secret "Database password" '^[A-Za-z0-9]{12,64}$' "That is not the 12 or more letters and digits you chose in Step 15." twice
      DB_PASS="$SECRET"
    fi
    if [ -z "$GEMINI" ]; then
      note "2) Gemini key: the long text in your .env file (starts with AIza)."
      ask_secret "Gemini key" '^[A-Za-z0-9._-]{20,200}$' "That does not look like a Gemini key (20 or more letters, digits, - or _)."
      GEMINI="$SECRET"
    fi
    if [ -z "$MANAGER" ]; then
      note "3) Manager password for the manager sign-in: 8 to 64 characters, only letters, digits and _ - . ! @ # % ^ * + ="
      ask_secret "Manager password" '^[A-Za-z0-9_.!@#%^*+=-]{8,64}$' "Use 8 to 64 letters, digits or _ - . ! @ # % ^ * + = (no spaces)." twice
      MANAGER="$SECRET"
    fi
  fi
  write_env
  SECRET=""
  ok "settings saved in $ENV_FILE (readable by root only, not part of the image or of Git)"
  note "bucket $BUCKET, region $R, database $DB_NAME on $DB_HOST, models are fetched from S3 at start"
}

# ------------------------------------------------------------------ 4 the image
stage_build() {
  say "4 of 6  Building the app image"
  note "The first build takes 8 to 12 minutes: it installs the packages and builds the vector store."
  note "A line appears every $BEAT seconds so you can see that it is alive. Keep this window open."
  local pid rc started now last
  started="$(date +%s)"
  docker build -t "$IMAGE" "$SRC" > "$BUILD_LOG" 2>&1 &
  pid=$!
  trap 'kill "$pid" 2>/dev/null; printf "\n   Stopped. Run the same command again to continue; finished build steps are kept.\n"; exit 130' INT TERM
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$BEAT"
    now="$(date +%s)"
    last="$(tr '\r' '\n' < "$BUILD_LOG" | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c1-70)"
    printf '   ... %dm%02ds  %s\n' $(( (now - started) / 60 )) $(( (now - started) % 60 )) "$last"
  done
  wait "$pid"; rc=$?
  trap - INT TERM
  if [ "$rc" -ne 0 ]; then
    printf '\n   The last lines of the build log:\n'
    tail -n 40 "$BUILD_LOG" | sed 's/^/   | /'
    die "The image could not be built (exit code $rc). The complete log is in $BUILD_LOG"
  fi
  docker image prune -f >/dev/null 2>&1
  ok "image $IMAGE built in $(( ($(date +%s) - started) / 60 )) min $(( ($(date +%s) - started) % 60 )) s"
}

# ------------------------------------------------------------------ 5 the container
stage_run() {
  say "5 of 6  Starting the app"
  local out running
  docker rm -f "$NAME" >/dev/null 2>&1
  out="$(docker run -d --name "$NAME" --restart unless-stopped -p "$PORT:8501" --env-file "$ENV_FILE" \
         --log-opt max-size=10m --log-opt max-file=3 "$IMAGE" 2>&1)" \
    || die "The container could not be started: $out"
  ok "container $NAME started (it starts again by itself after a reboot)"
  printf '   Waiting for the app to answer '
  for _ in $(seq 1 "$TRIES"); do
    if curl -fsS -m 3 "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
      printf ' ready\n'
      ok "the app answers on port $PORT"
      return 0
    fi
    running="$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" || running=""
    if [ "$running" != "true" ]; then
      printf '\n   The container stopped. Its last lines:\n'
      docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/   | /'
      die "The app did not start."
    fi
    printf '.'
    sleep 3
  done
  printf '\n   The last lines of the app log:\n'
  docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/   | /'
  die "The app did not answer in time (3 minutes)."
}

# ------------------------------------------------------------------ 6 the check and the address
public_ip() {
  local token
  token="$(curl -fsS -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null)" || return 0
  curl -fsS -m 3 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true
}

stage_check() {
  say "6 of 6  Deployment check (it runs inside the container, as the app does)"
  local rc ip
  docker exec "$NAME" python -m scripts.check_deploy
  rc=$?
  ip="$(public_ip)"
  printf '\n'
  if [ "$rc" -eq 0 ]; then
    printf '   Finished. The app is running.\n'
  else
    printf '   Finished, but the check above shows [FAIL] lines. Send a screenshot of this screen to Claude.\n'
  fi
  if [ -n "$ip" ]; then
    printf '\n   Open this address in your browser:   http://%s:%s\n' "$ip" "$PORT"
  else
    printf '\n   Open  http://<the fixed address of the server>:%s  in your browser.\n' "$PORT"
  fi
  printf '\n   Useful commands on the server:\n'
  printf '     sudo docker logs --tail 50 acko                    the last lines the app wrote\n'
  printf '     sudo docker restart acko                           restart the app (models are fetched from S3 again)\n'
  printf '     sudo docker exec acko python -m scripts.check_deploy   run the deployment check again\n'
  printf '     cd /opt/acko/src && sudo git pull && sudo bash deploy/ec2_deploy.sh   update after a new commit\n\n'
  return "$rc"
}

stage_server
stage_database
stage_secrets
stage_build
stage_run
stage_check
