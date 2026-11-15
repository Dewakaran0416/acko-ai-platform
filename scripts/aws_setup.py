"""Small helper for the AWS part. It never talks to Amazon: it only prints text or writes files on this PC.

    python -m scripts.aws_setup policy BUCKET   prints the permission policy for the IAM user (and copies it to the clipboard on Windows)
    python -m scripts.aws_setup keys BUCKET     saves the access key of the IAM user outside the project folder
                                                and writes S3_BUCKET, AWS_PROFILE and AWS_REGION into the .env file

The secret access key is typed hidden, is never printed, and is never written inside the project folder.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from acko import config

ENV_FILE = config.ROOT / ".env"
PROFILE = "acko-dev"
BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")        # 3 to 63 characters, no dots (dots break the https check)
PROFILE_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")
KEY_ID_RE = re.compile(r"AKIA[A-Z0-9]{16}")
SECRET_RE = re.compile(r"[A-Za-z0-9/+=]{40}")


class SetupError(Exception):
    """A mistake the user can fix; str(error) says how."""


# ------------------------------------------------------------------------------------------------ the policy
def clean_bucket(text) -> str:
    name = str(text or "").strip().strip("\"'")
    if not name:
        raise SetupError("Give the bucket name, for example:  python -m scripts.aws_setup policy acko-claims-yourname-2026")
    if not BUCKET_RE.fullmatch(name):
        raise SetupError(f"'{name[:70]}' is not a bucket name this project can use. Use 3 to 63 small letters, digits and "
                         "hyphens only (no capitals, no spaces, no dots, no underscores), starting and ending with a letter or digit.")
    return name


def policy_for(bucket: str) -> dict:
    """The least permission the app needs: list this one bucket, and read, write and delete the files in it. Nothing else."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ListThisBucket", "Effect": "Allow", "Action": "s3:ListBucket",
             "Resource": f"arn:aws:s3:::{bucket}"},
            {"Sid": "ReadWriteFilesInThisBucket", "Effect": "Allow",
             "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
        ],
    }


def copy_to_clipboard(text: str) -> bool:
    """Windows only: put the text on the clipboard with the built-in clip program. False when that is not possible."""
    if not sys.platform.startswith("win"):
        return False
    try:
        done = subprocess.run(["clip"], input=text.encode("ascii"), capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError, UnicodeEncodeError):
        return False
    return done.returncode == 0


def cmd_policy(args) -> int:
    bucket = clean_bucket(args.bucket or config.S3_BUCKET)
    text = json.dumps(policy_for(bucket), indent=2)
    print(f"Permission policy for the bucket {bucket}:\n")
    print(text)
    print()
    if copy_to_clipboard(text):
        print("It is on your clipboard now: paste it into the JSON box of the AWS page (Ctrl+V).")
    else:
        print("Select and copy the lines above, from the first curly bracket to the last one, into the JSON box of the AWS page.")
    return 0


# ------------------------------------------------------------------------------------------------ files
def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SetupError(f"The file {path} could not be read as text.")


def upsert_section(text: str, header: str, values: dict, drop: tuple = ()) -> str:
    """text with the section [header] holding the given values. Other sections, comments and other keys stay as they are."""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    new = [f"{name} = {value}" for name, value in values.items()]
    start = next((i for i, line in enumerate(lines) if line.strip() == f"[{header}]"), None)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        return "\n".join(lines + [f"[{header}]"] + new) + "\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    gone = {name.lower() for name in list(values) + list(drop)}
    kept = [line for line in lines[start + 1:end] if line.split("=", 1)[0].strip().lower() not in gone]
    return "\n".join(lines[:start + 1] + new + kept + lines[end:]) + "\n"


def set_env_lines(text: str, values: dict) -> str:
    """text (the content of a .env file) with NAME=value lines for the given names. Every other line stays exactly as it is."""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    out, done = [], set()
    for line in lines:
        found = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        name = found.group(1) if found else None
        if name in values:
            if name not in done:
                out.append(f"{name}={values[name]}")
                done.add(name)
        else:
            out.append(line)
    out += [f"{name}={value}" for name, value in values.items() if name not in done]
    return "\n".join(out) + "\n"


def credentials_path() -> Path:
    return Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or Path.home() / ".aws" / "credentials")


def config_path() -> Path:
    return Path(os.environ.get("AWS_CONFIG_FILE") or Path.home() / ".aws" / "config")


def save_profile(profile: str, key_id: str, secret: str, region: str, cred_file: Path | None = None, conf_file: Path | None = None):
    """Store the key in the AWS credentials file of this user (outside the project) as the named profile."""
    cred_file, conf_file = cred_file or credentials_path(), conf_file or config_path()
    cred_file.parent.mkdir(parents=True, exist_ok=True)
    conf_file.parent.mkdir(parents=True, exist_ok=True)
    cred_file.write_text(upsert_section(read_text(cred_file), profile,
                                        {"aws_access_key_id": key_id, "aws_secret_access_key": secret},
                                        drop=("aws_session_token",)), encoding="utf-8")
    try:
        os.chmod(cred_file, 0o600)               # only this user may read it (no effect on Windows, where the user folder is private)
    except OSError:
        pass
    header = "default" if profile == "default" else f"profile {profile}"
    conf_file.write_text(upsert_section(read_text(conf_file), header, {"region": region}), encoding="utf-8")
    return cred_file, conf_file


def update_env(values: dict, path: Path | None = None) -> Path:
    path = path or ENV_FILE
    path.write_text(set_env_lines(read_text(path), values), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------ the key
def ask_key_id() -> str:
    return input("Access key ID (starts with AKIA): ")


def ask_secret() -> str:
    return getpass.getpass("Secret access key (nothing is shown while you paste it; press Enter): ")


def tidy(value) -> str:
    """What was typed or pasted, without spaces, line breaks or quote marks around it."""
    return re.sub(r"\s+", "", str(value or "")).strip("\"'")


def check_pair(key_id: str, secret: str) -> None:
    if not key_id or not secret:
        raise SetupError("Both values are needed: the access key ID and the secret access key.")
    if key_id.startswith("ASIA"):
        raise SetupError("This is a temporary key (it starts with ASIA). Create an access key for the IAM user "
                         "acko-dev instead; its ID starts with AKIA.")
    if not KEY_ID_RE.fullmatch(key_id):
        raise SetupError(f"The access key ID should have 20 characters and start with AKIA; what you typed has {len(key_id)} "
                         "characters. Nothing was saved.")
    if not SECRET_RE.fullmatch(secret):
        raise SetupError(f"The secret access key should have 40 characters; what you pasted has {len(secret)}. "
                         "Copy it again with the copy icon on the AWS page. Nothing was saved.")


def cmd_keys(args) -> int:
    bucket = clean_bucket(args.bucket)
    profile = args.profile
    if not PROFILE_RE.fullmatch(profile):
        raise SetupError("The profile name may only have letters, digits, hyphens and underscores.")
    region = args.region
    print(f"Saving the access key for the bucket {bucket} (profile {profile}, region {region}).")
    print("Keep the AWS page with the new access key open, then type or paste the two values here.\n")
    key_id = tidy(ask_key_id())
    secret = tidy(ask_secret())
    check_pair(key_id, secret)
    cred_file, conf_file = save_profile(profile, key_id, secret, region)
    env_file = update_env({"S3_BUCKET": bucket, "AWS_PROFILE": profile, "AWS_REGION": region})
    print("\nSaved.")
    print(f"  access key : {cred_file}  (profile [{profile}], key ID ending {key_id[-4:]})")
    print("               This file is in your user folder, outside the project, so Git can never see it.")
    print(f"  region     : {conf_file}")
    print(f"  settings   : {env_file.name} now has S3_BUCKET={bucket}, AWS_PROFILE={profile}, AWS_REGION={region}")
    print("\nNext:  python -m scripts.test_storage")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.aws_setup", description="Helper for the AWS part of the project.")
    sub = parser.add_subparsers(dest="command", required=True)
    policy = sub.add_parser("policy", help="print the permission policy for the IAM user")
    policy.add_argument("bucket", nargs="?", default="", help="the bucket name (default: S3_BUCKET from .env)")
    policy.set_defaults(func=cmd_policy)
    keys = sub.add_parser("keys", help="save the access key outside the project and set up the .env file")
    keys.add_argument("bucket", help="the bucket name")
    keys.add_argument("--profile", default=PROFILE, help=f"name for the saved key (default: {PROFILE})")
    keys.add_argument("--region", default=config.AWS_REGION, help=f"region of the bucket (default: {config.AWS_REGION})")
    keys.set_defaults(func=cmd_keys)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SetupError as err:
        print(f"\nProblem: {err}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nStopped. Nothing was saved.")
        return 1
    except OSError as err:
        print(f"\nProblem: a file could not be written ({err.strerror or 'disk error'}). Nothing was changed after that point.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
