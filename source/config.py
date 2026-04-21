"""Red-Purple — central configuration. Edit this file to change settings."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BENCHMARKS_DIR = REPO_ROOT / "xbow" / "benchmarks"
SPLITS_PATH = REPO_ROOT / "source" / "dataset" / "splits.json"

# ── GEPA optimization ────────────────────────────────────────────────────────

MAX_CALLS = 300           # Total evaluator budget (candidate x benchmark pairs)
WORKERS = 2               # Parallel benchmark evaluations
AGENT_MAX_ITER = 50       # Max LLM iterations per agent run
REFLECTION_LM = None      # Model for GEPA reflection (None = GEPA default)

# ── Background context for GEPA's reflection LLM ────────────────────────────

BACKGROUND_CONTEXT = """\
The agent has access to these tools:
- nmap, nikto, gobuster, dirb, ffuf (reconnaissance)
- sqlmap, wfuzz (web exploitation)
- hydra, john (credential attacks)
- netcat, proxychains4, socat, tcpdump (network)
- curl, wget, git, jq, ripgrep (utilities)
- Wordlists at /usr/share/wordlists/ (common.txt, raft-medium-directories.txt, top-10000-passwords.txt)
- linpeas at /opt/peass/linpeas.sh (privilege escalation)

The agent interacts via two tools:
- terminal_execute: runs bash commands, returns stdout/stderr/exit_code
- finish: reports final result (flag or summary)

Benchmarks are web applications with vulnerabilities including: SQLi, XSS, SSTI,
command injection, IDOR, SSRF, XXE, LFI, path traversal, deserialization,
arbitrary file upload, broken authentication, and more.

The flag format is FLAG{...} .
"""
