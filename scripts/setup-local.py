"""Run with Python3.10+ on macOS or Windows. Downloads dependencies, no model weights."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--training", action="store_true", help="Install local PyTorch, Parquet and experiment dependencies")
args = parser.parse_args()
if sys.version_info < (3, 10):
    raise SystemExit("Install Python3.10 or newer")
node, npm = shutil.which("node"), shutil.which("npm.cmd" if os.name == "nt" else "npm")
if not node or not npm:
    raise SystemExit("Install Node.js22 LTS (including npm), then rerun this command")
subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")], check=True)
python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
if args.training and platform.system() == "Windows":
    subprocess.run([str(python), "-m", "pip", "install", "torch>=2.4,<3", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
extras = ".[training,test]" if args.training else ".[test]"
subprocess.run([str(python), "-m", "pip", "install", "-e", extras], cwd=root, check=True)
if os.name == "nt":
    npm_cli = Path(npm).parent / "node_modules/npm/bin/npm-cli.js"
    if not npm_cli.is_file():
        raise SystemExit("Cannot locate npm CLI next to npm.cmd; use the official Node.js Windows installation")
    npm_command = [node, str(npm_cli)]
else:
    npm_command = [npm]
subprocess.run([*npm_command, "ci"], cwd=root, check=True)
subprocess.run([*npm_command, "ci", "--prefix", "web"], cwd=root, check=True)
subprocess.run([*npm_command, "run", "engine:build"], cwd=root, check=True)
subprocess.run([str(python), "-m", "ptcg_lab.cli", "doctor"], cwd=root, check=True)
