#!/usr/bin/env python3
"""Generate a self-contained HTML prompt viewer for a red-purple experiment.
Usage: python visualize.py [experiment_dir]
Output: <experiment_dir>/viewer.html

Implementation lives in source/result-analysis/visualize/ (split into
data-loading modules and HTML/CSS/JS assets for maintainability).
"""
import subprocess
import sys
from pathlib import Path

_MAIN = Path(__file__).resolve().parent / "source" / "result-analysis" / "visualize" / "main.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(_MAIN), *sys.argv[1:]]))
