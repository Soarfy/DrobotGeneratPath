"""Bootstrap the pip-installed Isaac Sim by retrying until extensions converge.

Each Kit launch downloads and unpacks a batch of extensions, then can fail on
the one whose temporary zip Windows refused to delete. The unpacked copies are
cached, so a later launch proceeds further. Repeating "launch -> repair links"
eventually walks through the whole dependency set.

    python scripts/bootstrap_isaac.py --attempts 12

Stops as soon as the smoke test prints ISAAC_READY.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PY = Path(r"D:\SoftWare\Anaconda\envs\isaacsim\python.exe")
SMOKE = ROOT / "scripts" / "isaac_smoke.py"
FIXER = ROOT / "scripts" / "fix_isaac_links.py"


def count_cache() -> int:
    ov = Path(os.environ["LOCALAPPDATA"]) / "ov/data/exts/v2"
    if not ov.is_dir():
        return 0
    return sum(
        1 for p in ov.iterdir() if p.is_dir() and p.name not in ("index", "urls")
    )


def run_fixer() -> None:
    subprocess.run(
        [sys.executable, str(FIXER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=900, help="seconds per attempt")
    args = parser.parse_args()

    if not ISAAC_PY.exists():
        print(f"Isaac Python not found: {ISAAC_PY}")
        return 1

    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    tmp = Path("D:/temp_isaac")
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(tmp)

    prev_cache = count_cache()
    print(f"extensions cached at start: {prev_cache}")

    for attempt in range(1, args.attempts + 1):
        print(f"\n=== attempt {attempt}/{args.attempts} ===", flush=True)
        log_path = ROOT / "out" / f"isaac_boot_{attempt}.log"
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.run(
                    [str(ISAAC_PY), str(SMOKE)],
                    cwd=str(ROOT),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -1
            print("  timed out")

        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "ISAAC_READY" in text:
            print("\nIsaac Sim started successfully.")
            return 0

        failed = [
            line.split("Failed to pull extension:")[1].strip()
            for line in text.splitlines()
            if "Failed to pull extension:" in line
        ]
        cache_now = count_cache()
        print(f"  exit={code}  cache={cache_now} (+{cache_now - prev_cache})")
        if failed:
            print(f"  blocker: {failed[0][:110]}")
        prev_cache = cache_now

        if not failed and cache_now == prev_cache:
            print("  no progress and no clear blocker; stopping")
            return 2

        run_fixer()

    print("\nStill not starting. See out/isaac_boot_*.log.")
    print("Installing Isaac Sim from the Omniverse Launcher is the reliable path.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
