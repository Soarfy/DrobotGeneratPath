"""Pre-download a Kit extension bundle into the Omniverse cache, resumably.

Kit's own downloader is not resumable: if the connection drops it throws away
the partial zip and the next launch starts from zero. The Isaac ML archive is
3.1 GB, which on a slow link never completes. This script fetches it with HTTP
range resume, unpacks it into the same cache folder Kit uses, and creates the
junction Kit expects, so the next launch skips the download entirely.

    python scripts/prefetch_isaac_ext.py --url <zip-url> --ext-id omni.isaac.ml_archive-3.0.4+107.3.3.wx64.cp311

Get the exact URL and extension id from the Kit log line::

    Pulling extension: `omni.isaac.ml_archive-3.0.4` from the registry.
    downloading: 'https://.../omni.isaac.ml_archive-3.0.4%2B107.3.3.wx64.cp311.zip' -> ...

The script prints the URL it resolves from the log automatically when
``--url`` is omitted and a log file is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


CACHE = Path(os.environ["LOCALAPPDATA"]) / "ov/data/exts/v2"


def download_with_resume(url: str, dest: Path, retries: int = 200) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have and resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as fh:
                    last_report = time.time()
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - last_report > 10:
                            pct = f" ({have / total * 100:.1f}%)" if total else ""
                            print(
                                f"  {have / 1e6:.0f} MB{pct}"
                                f"  attempt {attempt}",
                                flush=True,
                            )
                            last_report = time.time()
            if total is None or part.stat().st_size >= total:
                part.replace(dest)
                return dest
            print(f"  incomplete ({part.stat().st_size}/{total}); resuming")
        except Exception as exc:  # network hiccup: keep the partial file
            print(f"  attempt {attempt} failed: {exc}; retrying in 5 s", flush=True)
            time.sleep(5)
    raise RuntimeError("download did not complete")


def unpack(zip_path: Path, ext_id: str) -> Path:
    if ext_id.lower().endswith(".zip"):
        ext_id = ext_id[:-4]
    target = CACHE / ext_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    print(f"unpacking into {target} ...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)

    # Some archives wrap everything in a top-level folder; flatten it so
    # extension.toml sits at the root as Kit expects.
    entries = list(target.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not (target / "extension.toml").exists():
        inner = entries[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(target / item.name))
        inner.rmdir()
    return target


def kit_bundle_dir() -> Path:
    return (
        Path(os.environ.get("ISAACSIM_ROOT", r"D:\SoftWare\Anaconda\envs\isaacsim"))
        / "Lib/site-packages/isaacsim/kit/data/kit/Isaac-Sim Python/5.1/exts/3"
    )


def link_into_bundle(target: Path, ext_id: str, dry_run: bool) -> None:
    bundle = kit_bundle_dir()
    if not bundle.is_dir():
        print(f"kit bundle not found: {bundle}")
        return
    link = bundle / ext_id
    if link.exists() and any(link.iterdir()):
        print(f"link already present: {link.name}")
        return
    if link.is_symlink() or link.exists():
        try:
            os.rmdir(link)
        except OSError:
            pass
    if dry_run:
        print(f"would link {link}")
        return
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode == 0:
        print(f"linked: {link.name}")
    else:
        print(f"link failed: {(result.stderr or result.stdout).strip()}")


def find_url_in_log(log_path: Path) -> tuple[str, str] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        m = re.search(r"downloading: '([^']+)' -> '[^']*?([^\\/']+\.zip)'", line)
        if m:
            url, filename = m.group(1), m.group(2)
            return url, filename
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--ext-id", default=None,
                        help="cache folder name, e.g. omni.x-1.2.3+build.wx64.cp311")
    parser.add_argument("--log", default=None, help="Kit log to read the URL from")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    url, ext_id = args.url, args.ext_id
    if (not url or not ext_id) and args.log:
        found = find_url_in_log(Path(args.log))
        if found:
            url, filename = found
            ext_id = ext_id or Path(filename).name
            print(f"from log: {ext_id}")
    if not url or not ext_id:
        print("need --url and --ext-id (or --log to read them)")
        return 1

    print(f"url    : {url}")
    print(f"ext id : {ext_id}")
    print(f"cache  : {CACHE}")

    zip_path = CACHE.parent / "downloads" / f"{ext_id}.zip"
    if not zip_path.exists():
        print(f"downloading to {zip_path}")
        download_with_resume(url, zip_path)
    else:
        print(f"already downloaded: {zip_path}")

    target = unpack(zip_path, ext_id)
    link_into_bundle(target, ext_id[:-4] if ext_id.lower().endswith(".zip") else ext_id, args.dry_run)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)

    print("\ndone. Launch Isaac Sim again; it should skip this download.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
