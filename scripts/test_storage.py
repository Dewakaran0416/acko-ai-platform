"""Checks the file storage from end to end: checks that the bucket is in the region set in .env, saves a small test photo
and a test PDF under the real key names, reads them back, lists them, checks that they are encrypted and private, and
deletes them again.

    python -m scripts.test_storage            run the test
    python -m scripts.test_storage --keep     leave the two test files in place (to look at them in the S3 console)

With S3_BUCKET empty the test runs on the local folder uploads/ (nothing is sent anywhere).
"""
from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
import uuid

from acko import config, storage
from acko.storage import StorageError

TEST_USER = "test-user"


class Skipped(Exception):
    """A check that could not be made (not a failure)."""


def tiny_photo() -> bytes:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (240, 160), (214, 226, 240))
    ImageDraw.Draw(image).rectangle((30, 40, 210, 120), fill=(180, 40, 40))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80)
    return out.getvalue()


def tiny_pdf() -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def bucket_region():
    """The region the bucket really lives in (S3 names it in its answer); None when S3 did not say."""
    try:
        headers = storage._client().head_bucket(Bucket=config.S3_BUCKET)["ResponseMetadata"]["HTTPHeaders"]
    except Exception:
        return None
    return headers.get("x-amz-bucket-region")


def public_status(key: str, region: str | None = None):
    """HTTP status of an anonymous request for the file (403 means private, as it should be); None when it could not be asked.

    The web address must name the region the bucket really lives in. For any other region S3 answers 301 (go to the other
    region) before it looks at the file, and that answer says nothing about whether the file is private."""
    url = f"https://{config.S3_BUCKET}.s3.{region or config.AWS_REGION}.amazonaws.com/{key}"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as err:
        return err.code
    except Exception:
        return None


class Report:
    def __init__(self):
        self.failed: list[str] = []
        self.count = 0

    def run(self, name: str, action) -> bool:
        self.count += 1
        try:
            note = action()
        except Skipped as why:
            self.count -= 1
            print(f"  [skip] {name}: {why}")
            return True
        except StorageError as err:
            print(f"  [FAIL] {name}: {err}")
        except AssertionError as err:
            print(f"  [FAIL] {name}: {err}")
        except Exception as err:
            print(f"  [FAIL] {name}: unexpected problem, {type(err).__name__}: {str(err)[:160]}")
        else:
            print(f"  [ ok ] {name}" + (f"  ({note})" if note else ""))
            return True
        self.failed.append(name)
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.test_storage", description="Test the file storage end to end.")
    parser.add_argument("--keep", action="store_true", help="do not delete the two test files at the end")
    args = parser.parse_args(argv)

    print("Storage test")
    print(f"  storage : {storage.describe()}")
    who = storage.identity()
    if who:
        print(f"  signed in as : {who}")
    print()
    report = Report()

    def usable():
        storage.check()
        return "the bucket was found and answers" if storage.backend() == "s3" else "the folder is ready"

    if not report.run("storage can be used", usable):
        print("\nResult: FAILED. Fix the point above first, then run this test again.")
        return 1

    claim_id = "t" + uuid.uuid4().hex[:12]
    photo_key, form_key = storage.photo_key(TEST_USER, claim_id), storage.form_key(TEST_USER, claim_id)
    try:
        photo, pdf = tiny_photo(), tiny_pdf()
    except Exception as err:
        print(f"The two small test files could not be made ({type(err).__name__}: {str(err)[:120]}).")
        print("Run: pip install pillow pypdf")
        return 1
    on_s3 = storage.backend() == "s3"
    stored: list[str] = []

    def save(key, data, kind):
        def action():
            storage.put(key, data, kind)
            stored.append(key)
            return f"{len(data)} bytes as {key}"
        return action

    def same(key, data):
        def action():
            if storage.get(key) != data:
                raise AssertionError("the file that came back is different from the one that was saved")
            return "identical"
        return action

    def listed():
        found = storage.list_keys(f"claims/{TEST_USER}/{claim_id}/")
        if found != sorted([photo_key, form_key]):
            raise AssertionError(f"expected both files, found {found}")
        return "both files are listed"

    def encrypted():
        if not on_s3:
            raise Skipped("only on S3")
        mode = storage.details(photo_key)["encryption"]
        if mode != "AES256":
            raise AssertionError(f"the file is not encrypted on S3 (encryption: {mode})")
        return "AES256 (SSE-S3)"

    def region_matches():
        if not on_s3:
            raise Skipped("only on S3")
        real = bucket_region()
        if real is None:
            raise Skipped("S3 did not say where the bucket is")
        if real != config.AWS_REGION:
            raise AssertionError(f"the bucket is in {real}, but AWS_REGION in the .env file is {config.AWS_REGION}. "
                                 f"Open .env, change that line to AWS_REGION={real}, and run this test again")
        return real

    def private():
        if not on_s3:
            raise Skipped("only on S3")
        code = public_status(photo_key, bucket_region() or config.AWS_REGION)
        if code is None:
            raise Skipped("the web address could not be asked")
        if code in (200, 206):
            raise AssertionError("ANYONE on the internet can open this file. In the S3 console open the bucket, tab Permissions, "
                                 "and switch Block all public access on")
        if code != 403:
            raise Skipped(f"the web address answered HTTP {code}, which does not show whether the file is private")
        return f"an anonymous request is refused (HTTP {code})"

    try:
        report.run("bucket is in the region of .env", region_matches)
        report.run("save the photo", save(photo_key, photo, storage.JPEG))
        report.run("save the form", save(form_key, pdf, storage.PDF))
        report.run("list the files", listed)
        report.run("read the photo back", same(photo_key, photo))
        report.run("read the form back", same(form_key, pdf))
        report.run("stored encrypted", encrypted)
        report.run("not open to the public", private)
    finally:
        if args.keep:
            print(f"\n  kept for you to look at:\n    {photo_key}\n    {form_key}")
        else:
            def clean():
                for key in stored:
                    storage.remove(key)
                left = [key for key in stored if storage.exists(key)]
                if left:
                    raise AssertionError(f"still there: {left}")
                return "test files deleted"
            report.run("delete the test files", clean)

    print()
    if report.failed:
        print(f"Result: FAILED ({len(report.failed)} of {report.count}): " + ", ".join(report.failed))
        return 1
    print(f"Result: ALL OK ({report.count} of {report.count} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
