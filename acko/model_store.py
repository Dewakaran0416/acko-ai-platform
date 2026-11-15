"""The trained model files on Amazon S3 (the brief: joblib .pkl files stored on AWS S3).

    python -m scripts.sync_models upload     copy models/*.pkl from this computer to s3://BUCKET/models/
    python -m scripts.sync_models status     compare models/ with the S3 copies (changes nothing)
    python -m scripts.sync_models download   make models/ match S3

The web app calls download() once when it starts on the AWS server (MODELS_FROM_S3=true in the server's settings),
so the server always runs the models that are on S3. The copy inside the Docker image is the fallback when S3
cannot be reached: the app still starts and says so. On your own computer nothing is downloaded unless you run
the command above, so a model you have just trained is never overwritten by an older copy from S3.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from acko import config, storage

PREFIX = (config.env("S3_MODEL_PREFIX") or "models").strip("/")        # folder name inside the bucket
MODEL_NAMES = ("quote_model.pkl", "claim_car_amount.pkl", "claim_car_approval.pkl",
               "claim_bike_amount.pkl", "claim_bike_approval.pkl")      # what the web app loads (see quote.py, claims.py)
STOP_KINDS = {"credentials", "network", "region", "bucket", "access", "clock"}   # if one file fails like this, all will


@dataclass
class Item:
    name: str
    result: str        # uploaded, downloaded, same, different, local-only, s3-only, missing, failed, skipped
    detail: str = ""
    size: int = 0      # bytes of the file that is now in use (or of the S3 copy for s3-only)


def wanted() -> bool:
    """True when this process should fetch the models from S3 when it starts (the AWS server sets MODELS_FROM_S3=true)."""
    return config.env("MODELS_FROM_S3").lower() in ("1", "true", "yes", "on")


def key_of(name: str) -> str:
    return f"{PREFIX}/{name}"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _head(key: str) -> dict | None:
    """What S3 knows about a key ({'size', 'etag'}), or None when it is not there. Raises StorageError for real problems."""
    try:
        head = storage._client().head_object(Bucket=config.S3_BUCKET, Key=key)
    except Exception as exc:
        problem = storage._explain(exc, key)
        if problem.kind == "missing":
            return None
        raise problem from exc
    return {"size": int(head.get("ContentLength", 0)), "etag": str(head.get("ETag", "")).strip('"')}


def _need_s3() -> None:
    if storage.backend() != "s3":
        raise storage.StorageError("S3_BUCKET is not set, so there is no bucket to use. Set it in the .env file.", "bucket")


def _local_names() -> list[str]:
    found = {path.name for path in config.MODEL_DIR.glob("*.pkl")} if config.MODEL_DIR.is_dir() else set()
    return sorted(found | set(MODEL_NAMES))


def where() -> str:
    return f"s3://{config.S3_BUCKET}/{PREFIX}/" if storage.backend() == "s3" else "(S3_BUCKET is not set)"


# ------------------------------------------------------------------------------------------------ the three actions
def status() -> list[Item]:
    """Compare models/ with S3 without changing anything. Raises StorageError when S3 cannot be used."""
    _need_s3()
    items = []
    for name in _local_names():
        local = config.MODEL_DIR / name
        head = _head(key_of(name))
        if local.is_file() and head is None:
            items.append(Item(name, "local-only", "not on S3 yet", local.stat().st_size))
        elif not local.is_file() and head is None:
            items.append(Item(name, "missing", "neither here nor on S3"))
        elif not local.is_file():
            items.append(Item(name, "s3-only", "only on S3", head["size"]))
        elif _md5(local) == head["etag"]:
            items.append(Item(name, "same", "identical to the S3 copy", local.stat().st_size))
        else:
            items.append(Item(name, "different", f"local {local.stat().st_size:,} bytes, S3 {head['size']:,} bytes", local.stat().st_size))
    return items


def upload() -> list[Item]:
    """Copy every models/*.pkl to S3 (a file whose S3 copy is already identical is skipped). Raises StorageError."""
    _need_s3()
    items = []
    for path in sorted(config.MODEL_DIR.glob("*.pkl")):
        key = key_of(path.name)
        head = _head(key)
        if head is not None and _md5(path) == head["etag"]:
            items.append(Item(path.name, "same", "already on S3", path.stat().st_size))
            continue
        storage.put(key, path.read_bytes(), "application/octet-stream")
        items.append(Item(path.name, "uploaded", f"-> {where()}{path.name} (encrypted)", path.stat().st_size))
    return items


def download(force: bool = False) -> list[Item]:
    """Make models/ match S3. Never raises: a problem is reported in the result and the local copy stays in use."""
    items: list[Item] = []
    try:
        _need_s3()
    except storage.StorageError as exc:
        return [Item("(all)", "skipped", str(exc))]
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stopped = ""
    for name in MODEL_NAMES:
        local = config.MODEL_DIR / name
        size = local.stat().st_size if local.is_file() else 0
        if stopped:
            items.append(Item(name, "skipped", "S3 could not be reached, the local copy is used", size))
            continue
        try:
            head = _head(key_of(name))
            if head is None:
                items.append(Item(name, "local-only" if local.is_file() else "missing",
                                  "not on S3" + (", the copy in the image is used" if local.is_file() else ""), size))
                continue
            if not force and local.is_file() and _md5(local) == head["etag"]:
                items.append(Item(name, "same", "identical to the S3 copy", size))
                continue
            data = storage.get(key_of(name))
            if len(data) != head["size"]:
                raise storage.StorageError("The download was cut short.", "network")
            handle = tempfile.NamedTemporaryFile(dir=config.MODEL_DIR, suffix=".part", delete=False)
            try:
                with handle:
                    handle.write(data)
                os.replace(handle.name, local)
            except OSError:
                Path(handle.name).unlink(missing_ok=True)
                raise
            items.append(Item(name, "downloaded", f"from {where()}{name}", len(data)))
        except storage.StorageError as exc:
            if exc.kind in STOP_KINDS:
                stopped = str(exc)
            items.append(Item(name, "failed", str(exc) + (", the local copy is used" if local.is_file() else ""), size))
        except OSError as exc:
            items.append(Item(name, "failed", f"The file could not be saved ({exc.strerror or 'disk error'})", size))
    return items


def describe(items: list[Item]) -> str:
    """One sentence for the page: where the models in use came from."""
    if not items:
        return ""
    got = sum(1 for item in items if item.result == "downloaded")
    same = sum(1 for item in items if item.result == "same")
    bad = sum(1 for item in items if item.result in ("failed", "skipped", "missing"))
    if items[0].result == "skipped" and items[0].name == "(all)":
        return "AI models: S3 is not set up, the local files are used"
    text = f"AI models: {got + same} of {len(items)} checked against Amazon S3"
    if got:
        text += f" ({got} downloaded)"
    if bad:
        text += f"; {bad} could not be checked, the copy in the app is used"
    return text
