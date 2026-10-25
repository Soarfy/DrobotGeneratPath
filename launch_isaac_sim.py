"""Isaac Sim launcher.

Isaac Sim ships its own Python interpreter and expects its Kit environment.
Run this file with that interpreter, not the project venv:

    Windows:
        C:\\isaac-sim\\python.bat launch_isaac_sim.py --trajectory out/trajectory.json
    Linux:
        ~/isaac-sim/python.sh launch_isaac_sim.py --trajectory out/trajectory.json

Options:
    --headless     run without the renderer (faster, no window)
    --realtime     advance at wall-clock speed instead of as fast as possible
    --config       robot/scene YAML (default config/robot.yaml)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--trajectory", default=str(ROOT / "out/trajectory.json"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    # Fail fast with a clear message instead of a Kit stack trace.
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        try:
            import omni.isaac  # noqa: F401
        except ImportError:
            print(
                "Isaac Sim Python modules not found.\n"
                "Run this script with Isaac Sim's own interpreter:\n"
                "  Windows: <ISAAC_SIM>\\python.bat launch_isaac_sim.py\n"
                "  Linux:   <ISAAC_SIM>/python.sh launch_isaac_sim.py",
                file=sys.stderr,
            )
            return 2

    from dobot_pathgen.backends.isaac_sim import run_isaac_sim
    from dobot_pathgen.config import Config

    cfg = Config.load(args.config)
    return run_isaac_sim(cfg, args.trajectory, headless=args.headless, realtime=args.realtime)


if __name__ == "__main__":
    sys.exit(main())
