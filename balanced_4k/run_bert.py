#!/usr/bin/env python3
"""Easy runner: balanced 4k + bert embedder. See fever_easy.py for the full explained code."""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
py = root / ".venv" / "bin" / "python"
cmd = [str(py if py.exists() else sys.executable), str(root / "fever_easy.py"),
       "--size", "4k", "--embedder", "bert", "--modes", "Default,Grid Search,Random Search,Bayesian Optimization"]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
