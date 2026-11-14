"""Where the files of a claim live: Amazon S3 when S3_BUCKET is set, otherwise the folder uploads/ on this computer.

The keys follow the brief and are the same in both places, so moving from the laptop to AWS changes nothing in the app:

    claims/{user_id}/{claim_id}/damage.jpg      the damage photo (a clean JPEG: location data removed by acko.vision)
    claims/{user_id}/{claim_id}/form.pdf        the claim form

On S3 every file is stored encrypted (SSE-S3) and no public access is ever set here: the bucket stays private.
Credentials are never in the code. On your PC they come from the profile saved by  python -m scripts.aws_setup keys
(named in AWS_PROFILE); on the EC2 server they come from its IAM role.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from acko import config

LOCAL_DIR = config.ROOT / "uploads"
MAX_BYTES = 12 * 1024 * 1024                 # a bit more than the biggest photo (10 MB) or form (5 MB) that the app accepts
KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*")   # no '..', no leading '/', no spaces
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
JPEG = "image/jpeg"
PDF = "application/pdf"

NO_KEY_TEXT = ("No AWS access key was found. On your PC run: python -m scripts.aws_setup keys YOUR-BUCKET-NAME   "
               "On the EC2 server the IAM role of the instance is used, so check that the role is attached to it.")


class StorageError(Exception):
    """A problem the user can understand: str(error) is the message to show. error.kind says what kind it was:
    key, credentials, access, bucket, region, network, clock, missing, size or other."""

    def __init__(self, message: str, kind: str = "other"):
        super().__init__(message)
        self.kind = kind


# ------------------------------------------------------------------------------------------------ keys
def _part(value, what: str) -> str:
    text = "" if value is None else str(value)
    if not ID_PATTERN.fullmatch(text):
        raise StorageError(f"The {what} '{text[:40]}' cannot be used in a file name.", "key")
    return text


def photo_key(user_id, claim_id) -> str:
    return f"claims/{_part(user_id, 'user id')}/{_part(claim_id, 'claim id')}/damage.jpg"


def form_key(user_id, claim_id) -> str:
    return f"claims/{_part(user_id, 'user id')}/{_part(claim_id, 'claim id')}/form.pdf"


def check_key(key: str) -> str:
    """The key itself, or a StorageError when it could point outside the storage (for example '../x' or '/etc/x')."""
    if not isinstance(key, str) or len(key) > 255 or not KEY_PATTERN.fullmatch(key):
        raise StorageError("This file name is not allowed.", "key")
    return key


# ------------------------------------------------------------------------------------------------ where
def backend() -> str:
    """'s3' when S3_BUCKET is set, otherwise 'local'."""
    return "s3" if config.S3_BUCKET else "local"


def describe() -> str:
    if backend() == "s3":
        return f"Amazon S3 bucket {config.S3_BUCKET} ({config.AWS_REGION}), files are stored encrypted"
    try:
        shown = str(LOCAL_DIR.relative_to(config.ROOT))
    except ValueError:
        shown = str(LOCAL_DIR)
    return f"local folder {shown} (S3_BUCKET is not set)"


def _local_path(key: str):
    path = (LOCAL_DIR / check_key(key)).resolve()
    if LOCAL_DIR.resolve() not in path.parents:
        raise StorageError("This file name is not allowed.", "key")
    return path


def _session():
    import boto3
    if not os.environ.get("AWS_PROFILE", "x").strip():
        del os.environ["AWS_PROFILE"]          # an empty 'AWS_PROFILE=' line (for example in .env) would make boto3 look for a profile without a name
    profile = config.env("AWS_PROFILE")
    return boto3.Session(profile_name=profile) if profile else boto3.Session()


@lru_cache(maxsize=1)
def _client():
    try:
        from botocore.config import Config
        session = _session()
    except ModuleNotFoundError as exc:
        raise StorageError("The package boto3 is not installed here. Run: pip install boto3", "other") from exc
    return session.client("s3", region_name=config.AWS_REGION,
                          config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2, "mode": "standard"}))


def _explain(exc: Exception, key: str | None = None) -> StorageError:
    """A plain-words StorageError for whatever boto3 raised. Secrets are never part of the message."""
    from botocore import exceptions as aws
    if isinstance(exc, StorageError):
        return exc
    bucket = config.S3_BUCKET
    if isinstance(exc, (aws.NoCredentialsError, aws.PartialCredentialsError, aws.ProfileNotFound, aws.CredentialRetrievalError)):
        return StorageError(NO_KEY_TEXT, "credentials")
    if isinstance(exc, aws.SSLError):
        return StorageError("A secure connection to Amazon S3 could not be made. Check the date and time of this computer "
                            "and any VPN, proxy or antivirus program that inspects HTTPS traffic.", "network")
    if isinstance(exc, (aws.ConnectionError, aws.HTTPClientError)):
        return StorageError("Amazon S3 could not be reached (internet connection or region problem). Please try again.", "network")
    if isinstance(exc, (aws.ParamValidationError, aws.InvalidRegionError)):
        return StorageError("S3_BUCKET or AWS_REGION in the .env file is not a valid name.", "bucket")
    if isinstance(exc, aws.ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code == "NoSuchBucket" or (code == "404" and key is None):
            return StorageError(f"The S3 bucket '{bucket}' was not found. Check S3_BUCKET (and AWS_REGION) in the .env file.", "bucket")
        if code in ("NoSuchKey", "404", "NotFound"):
            return StorageError(f"The file {key} was not found." if key else "The file was not found.", "missing")
        if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "AuthFailure", "UnrecognizedClientException", "InvalidClientTokenId"):
            return StorageError("AWS did not accept the access key. Create a new key for the IAM user and run: "
                                "python -m scripts.aws_setup keys YOUR-BUCKET-NAME", "credentials")
        if code in ("ExpiredToken", "ExpiredTokenException", "RequestExpired"):
            return StorageError("The AWS credentials have expired. Run: python -m scripts.aws_setup keys YOUR-BUCKET-NAME", "credentials")
        if code in ("AccessDenied", "403", "AllAccessDisabled", "Forbidden"):
            return StorageError(f"AWS refused the request: this user or role is not allowed to use the bucket '{bucket}'. "
                                "Check its permission policy (python -m scripts.aws_setup policy prints the right one).", "access")
        if code in ("PermanentRedirect", "301", "AuthorizationHeaderMalformed", "IllegalLocationConstraintException"):
            return StorageError(f"The bucket '{bucket}' is in another region than AWS_REGION ({config.AWS_REGION}). "
                                "Set AWS_REGION to the region of the bucket.", "region")
        if code == "RequestTimeTooSkewed":
            return StorageError("The clock of this computer is wrong by more than 15 minutes, so AWS refuses the request. "
                                "Set the date and time to automatic.", "clock")
        return StorageError(f"Amazon S3 answered with an error ({code or 'no code'}).", "other")
    return StorageError(f"The storage failed ({type(exc).__name__}).", "other")


# ------------------------------------------------------------------------------------------------ the actions
def check() -> str:
    """Make sure the storage can be used and return a one-line description. Raises StorageError with a plain explanation."""
    if backend() == "local":
        try:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"The folder for the uploads could not be created ({exc.strerror or 'disk error'}).", "other") from exc
        return describe()
    try:
        _client().head_bucket(Bucket=config.S3_BUCKET)
    except Exception as exc:
        raise _explain(exc) from exc
    return describe()


def put(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Save a file; returns the key (that is what the database stores)."""
    check_key(key)
    if not data:
        raise StorageError("The file is empty.", "size")
    if len(data) > MAX_BYTES:
        raise StorageError(f"The file is bigger than {MAX_BYTES // (1024 * 1024)} MB.", "size")
    if backend() == "local":
        try:
            path = _local_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as exc:
            raise StorageError(f"The file could not be saved on this computer ({exc.strerror or 'disk error'}).", "other") from exc
        return key
    try:
        _client().put_object(Bucket=config.S3_BUCKET, Key=key, Body=data, ContentType=content_type, ServerSideEncryption="AES256")
    except Exception as exc:
        raise _explain(exc, key) from exc
    return key


def get(key: str) -> bytes:
    check_key(key)
    if backend() == "local":
        path = _local_path(key)
        if not path.is_file():
            raise StorageError(f"The file {key} was not found.", "missing")
        return path.read_bytes()
    try:
        return _client().get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()
    except Exception as exc:
        raise _explain(exc, key) from exc


def exists(key: str) -> bool:
    check_key(key)
    if backend() == "local":
        return _local_path(key).is_file()
    try:
        _client().head_object(Bucket=config.S3_BUCKET, Key=key)
        return True
    except Exception as exc:
        problem = _explain(exc, key)
        if problem.kind == "missing":
            return False
        raise problem from exc


def details(key: str) -> dict:
    """{'size': bytes, 'encryption': ...}: 'AES256' for a file on S3 that is stored encrypted, 'local' on this computer."""
    check_key(key)
    if backend() == "local":
        path = _local_path(key)
        if not path.is_file():
            raise StorageError(f"The file {key} was not found.", "missing")
        return {"size": path.stat().st_size, "encryption": "local"}
    try:
        head = _client().head_object(Bucket=config.S3_BUCKET, Key=key)
    except Exception as exc:
        raise _explain(exc, key) from exc
    return {"size": int(head.get("ContentLength", 0)), "encryption": head.get("ServerSideEncryption", "none")}


def remove(key: str) -> None:
    """Delete a file (nothing happens when it is not there)."""
    check_key(key)
    if backend() == "local":
        try:
            _local_path(key).unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError(f"The file could not be deleted ({exc.strerror or 'disk error'}).", "other") from exc
        return
    try:
        _client().delete_object(Bucket=config.S3_BUCKET, Key=key)
    except Exception as exc:
        raise _explain(exc, key) from exc


def list_keys(prefix: str = "", limit: int = 1000) -> list[str]:
    """The keys that start with prefix (at most limit), sorted."""
    if prefix:
        check_key(prefix.rstrip("/") or "x")
    if backend() == "local":
        if not LOCAL_DIR.exists():
            return []
        base = LOCAL_DIR.resolve()
        found = (path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file())
        return sorted(key for key in found if key.startswith(prefix))[:limit]
    try:
        keys: list[str] = []
        for page in _client().get_paginator("list_objects_v2").paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
            if len(keys) >= limit:
                break
        return sorted(keys)[:limit]
    except Exception as exc:
        raise _explain(exc) from exc


def identity() -> str:
    """Who AWS thinks you are, without the account number (for example 'user/acko-dev'); '' on the local backend or when unknown."""
    if backend() != "s3":
        return ""
    try:
        arn = _session().client("sts", region_name=config.AWS_REGION).get_caller_identity()["Arn"]
        return arn.split(":", 5)[-1]
    except Exception:
        return ""
