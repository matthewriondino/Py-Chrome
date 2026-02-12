#!/usr/bin/env python3
"""
One-command application builder for distribution to non-Python users.

This script always installs/updates dependencies, then exports a release bundle
and zip using build_release.py.

Usage:
  python3 build_application.py
  python3 build_application.py --no-zip
  python3 build_application.py --clean-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE_SCRIPT = ROOT / "build_release.py"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install dependencies and build a shareable Py-Chrome application package."
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Build app/package but skip creating the release zip archive.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Delete build outputs and exit (no install/build).",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep PyInstaller build/ and dist/ folders after successful export.",
    )
    return parser.parse_args()


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed intermediate folder: {path}")


def main() -> int:
    args = parse_args()

    if not RELEASE_SCRIPT.exists():
        print(f"Missing required script: {RELEASE_SCRIPT}")
        return 1

    cmd = [sys.executable, str(RELEASE_SCRIPT)]

    # Always install deps for real builds, but skip for explicit clean-only runs.
    if not args.clean_only:
        cmd.append("--install-deps")
    if args.no_zip:
        cmd.append("--no-zip")
    if args.clean_only:
        cmd.append("--clean-only")
    printable = " ".join(cmd)
    print(f"Running: {printable}")

    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        if args.clean_only:
            print("\nClean finished.")
            return 0

        if not args.keep_intermediate:
            _remove_tree(BUILD_DIR)
            _remove_tree(DIST_DIR)

        print("\nDone.")
        print("Application package is ready in:")
        print(f"- {ROOT / 'release'}")
        if not args.keep_intermediate:
            print("Only the release folder is required for sharing/running.")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Build failed with exit code {exc.returncode}.")
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
