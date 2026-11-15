"""Models on Amazon S3: upload them from your PC, or see what the server will use.

    python -m scripts.sync_models status      compare models/ with s3://BUCKET/models/ (changes nothing)
    python -m scripts.sync_models upload      copy models/*.pkl to S3 (encrypted, private)
    python -m scripts.sync_models download    make models/ match S3 (the server does this by itself when it starts)

Needs S3_BUCKET in the .env file (see python -m scripts.aws_setup).
"""
from __future__ import annotations

import argparse

from acko import config, model_store, storage

ACTIONS = {"status": model_store.status, "upload": model_store.upload, "download": lambda: model_store.download(force=False)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Models on Amazon S3")
    parser.add_argument("action", choices=sorted(ACTIONS), help="status, upload or download")
    action = parser.parse_args().action

    if storage.backend() != "s3":
        print("FAIL  S3_BUCKET is not set. Put it in the .env file (python -m scripts.aws_setup does it for you).")
        return 1
    print(f"Bucket : {config.S3_BUCKET}  ({config.AWS_REGION})")
    print(f"Folder : {model_store.where()}")
    try:
        items = ACTIONS[action]()
    except storage.StorageError as exc:
        print("FAIL ", exc)
        return 1
    for item in items:
        print(f"   {item.name:<26} {item.result:<10} {item.size:>10,} bytes   {item.detail}")
    bad = [item for item in items if item.result in ("failed", "missing", "skipped")]
    if action == "status":
        different = [item for item in items if item.result in ("different", "local-only", "s3-only")]
        print("OK    Every model file is the same here and on S3." if not bad and not different else
              "NOTE  Some files differ. Run:  python -m scripts.sync_models upload   (your PC is newer)")
    elif bad:
        print(f"FAIL  {len(bad)} file(s) could not be processed.")
        return 1
    else:
        print(f"OK    {len(items)} model file(s) checked ({sum(1 for i in items if i.result in ('uploaded', 'downloaded'))} copied).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
