"""Repair the pip-installed Isaac Sim extension bundle on Windows.

Why this is needed
------------------
The pip wheel ships Kit's extension bundle as symlinks into
``%LOCALAPPDATA%\\ov\\data\\exts\\v2``. When Isaac Sim was installed from Git
Bash, those symlinks were written with Unix-style targets (``/c/Users/...``),
which Windows cannot resolve. Kit then logs ``Failed to pull extension`` and
finally dies with ``ModuleNotFoundError: No module named 'omni.kit.usd'``.

What it does
------------
1. Scans the Kit bundle folder for links that do not resolve.
2. For each one, finds the matching unpacked folder in the Omniverse cache and
   replaces the link with a native Windows directory junction.
3. For cached extensions that have no link entry at all, creates a junction
   using the cache's own naming, which Kit accepts.

Usage::

    python scripts/fix_isaac_links.py
    python scripts/fix_isaac_links.py --dry-run

No administrator rights are needed: directory junctions do not require them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def default_paths():
    kit = (
        Path(os.environ.get("ISAACSIM_ROOT", r"D:\SoftWare\Anaconda\envs\isaacsim"))
        / "Lib/site-packages/isaacsim/kit/data/kit/Isaac-Sim Python/5.1/exts/3"
    )
    ov = Path(os.environ["LOCALAPPDATA"]) / "ov/data/exts/v2"
    return kit, ov


def resolves(path: Path) -> bool:
    """True when the path points at something Windows can actually open."""
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def make_junction(link: Path, target: Path, dry_run: bool) -> bool:
    if link.exists() or link.is_symlink():
        if resolves(link):
            return True
        # Remove the broken link first. rmdir works for directory symlinks.
        if not dry_run:
            try:
                os.rmdir(link)
            except OSError:
                try:
                    link.unlink()
                except OSError as exc:
                    print(f"    cannot remove {link.name}: {exc}")
                    return False
    if dry_run:
        print(f"    would link {link.name} -> {target.name}")
        return True
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        print(f"    FAILED {link.name}: {msg}")
        return False
    return True


def version_from_kit_toml(cache_dir: Path) -> str | None:
    """Read the real version from the unpacked extension's extension.toml."""
    toml = cache_dir / "config" / "extension.toml"
    if not toml.is_file():
        toml = cache_dir / "extension.toml"
    if not toml.is_file():
        return None
    for line in toml.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            _, _, value = stripped.partition("=")
            return value.strip().strip('"')
    return None


def extension_name(entry: str) -> str:
    """'omni.kit.pip_archive-5df61bf515266ea2' -> 'omni.kit.pip_archive'."""
    # Extension names contain dots and underscores but the version/hash part
    # always starts at the first hyphen followed by a hex-ish token.
    head, sep, _ = entry.partition("-")
    return head if sep else entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", default=None, help="Kit bundle exts/3 folder")
    parser.add_argument("--ov", default=None, help="Omniverse extension cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kit_default, ov_default = default_paths()
    kit = Path(args.kit) if args.kit else kit_default
    ov = Path(args.ov) if args.ov else ov_default

    print(f"kit bundle : {kit}")
    print(f"ov cache   : {ov}")
    if not kit.is_dir():
        print("Kit bundle folder not found. Point --kit at the exts/3 directory.")
        return 1
    if not ov.is_dir():
        print("Omniverse cache not found. Start Isaac Sim once so it unpacks, then rerun.")
        return 1

    cached = {p.name: p for p in ov.iterdir() if p.is_dir() and p.name not in ("index", "urls")}
    by_ext: dict[str, Path] = {}
    for name, path in cached.items():
        by_ext.setdefault(extension_name(name), path)

    print(f"cached extensions: {len(cached)}")
    print()

    links = sorted(p for p in kit.iterdir() if p.name.startswith(("omni.", "carb.")))
    fixed = 0
    broken = 0
    for link in links:
        if resolves(link):
            continue
        broken += 1
        ext = extension_name(link.name)
        target = by_ext.get(ext)
        print(f"  broken: {link.name}")
        if target is None:
            print(f"    no cached copy for '{ext}' (will be downloaded on next launch)")
            continue
        if make_junction(link, target, args.dry_run):
            fixed += 1
            print(f"    linked -> {target.name}")

    # Extensions unpacked in the cache that have no link at all.
    linked_exts = {extension_name(p.name) for p in links}
    for ext, target in sorted(by_ext.items()):
        if ext in linked_exts:
            continue
        print(f"  missing link: {ext}")

        # Newer Kit builds unpack straight into a folder whose name IS the full
        # extension id ("omni.foo-1.2.3+<build>.wx64.cp311"). Reusing that name
        # verbatim is the only reliable option: rebuilding a version/ABI suffix
        # by hand produced links that Kit refused to load.
        if "+" in target.name:
            link_name = target.name
        else:
            version = version_from_kit_toml(target) or "0.0.0"
            sample = next((p.name for p in links), "")
            suffix = ""
            if "+" in sample:
                suffix = "+" + sample.split("+", 1)[1]
            link_name = f"{ext}-{version}{suffix}"

        new_link = kit / link_name
        if make_junction(new_link, target, args.dry_run):
            fixed += 1
            print(f"    created {new_link.name}")

    print()
    print(f"broken links: {broken}   repaired: {fixed}")
    if args.dry_run:
        print("dry run: nothing changed")
    else:
        print("Now run:  scripts\\test_isaac.cmd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
