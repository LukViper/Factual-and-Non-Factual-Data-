#!/usr/bin/env python3
"""Easy runner: balanced 67k + bert embedder. See fever_easy.py for the full explained code."""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
py = root / ".venv" / "bin" / "python"
cmd = [str(py if py.exists() else sys.executable), str(root / "fever_easy.py"),
       "--size", "67k", "--embedder", "bert", "--modes", "Default"]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
