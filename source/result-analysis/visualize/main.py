#!/usr/bin/env python3
"""Generate a self-contained HTML prompt viewer for a red-purple experiment.
Usage: python main.py [experiment_dir]
Output: <experiment_dir>/viewer.html
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

from data_loader import load_experiment
from render import render_html

# Standalone entrypoint — doesn't inherit the dotenv loading the main GEPA run
# does, so OPENROUTER_API_KEY (needed for the similarity heatmap) would only
# ever come from a real env var otherwise, even when it's set in repo-root .env.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/experiment1")
    data = load_experiment(exp_dir)
    html = render_html(data)
    out = exp_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
