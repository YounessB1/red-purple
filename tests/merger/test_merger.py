"""Test merger using fixtures copied from experiment13.

Calls _run_merge directly — bypasses the overlap/Pareto check entirely.

  Candidate A (seed, 3 files)        → workspace/agent/
  Candidate B (seed + 2 skills)      → workspace/agent_to_merge/
  Artifacts from fixtures/iter/train/parent/ → workspace/artifacts/

WARNING: overwrites workspace/ during the run (same as production).
"""

import shutil
import yaml
from pathlib import Path

from source.optimize_anything import candidate_store
from source.optimize_anything.agentic_reflector import AgenticReflector, _WORKSPACE, _WORKSPACE_AGENT
from source.optimize_anything.utils import folder_to_dict

_ROOT     = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures"
_OUT      = Path(__file__).parent / "output"

HASH_A = "002c1ae71324fd4a4384a9da2d522357771fe46a92d38d8ab3e5af57e75b78c2"
HASH_B = "bbe48d72a16645db8dbb3dd89910babd97789801f8f14076575e82a16b2b8c37"


class _Logger:
    def log_reflector(self, input_tokens, output_tokens, message, raw, cost=0.0, steps=None):
        print(f"[test-merger] tokens in={input_tokens} out={output_tokens} "
              f"cost=${cost:.4f} steps={steps}")
    def log_reflector_changes(self, changes):
        print(f"[test-merger] changes ({len(changes)} chars):")
        print(changes[:600] + ("..." if len(changes) > 600 else ""))


def main() -> None:
    cfg             = yaml.safe_load((_ROOT / "config.yaml").read_text())
    reflector_model = cfg["reflector"]["md"]["model"]
    reflector_agent = cfg["reflector"].get("agent", "reflector")
    merger_model    = cfg["merger"]["md"]["model"]
    merger_agent    = cfg["merger"].get("agent", "merger")
    merge_threshold = cfg["merger"].get("merge_threshold", 0.5)

    # Point candidate_store at our local fixtures
    candidate_store.configure(_FIXTURES / "candidates")

    print(f"[test-merger] A={HASH_A[:12]}…  B={HASH_B[:12]}…")
    print(f"[test-merger] A files: {sorted(candidate_store.load(HASH_A).keys())}")
    print(f"[test-merger] B files: {sorted(candidate_store.load(HASH_B).keys())}")

    # Clear workspace and restore candidate A into workspace/agent/
    if _WORKSPACE.exists():
        shutil.rmtree(_WORKSPACE)
    _WORKSPACE.mkdir(parents=True)
    candidate_store.restore_workspace(HASH_A, _WORKSPACE_AGENT)

    # Copy artifacts so the merger has failure context
    artifacts_src = _FIXTURES / "iter" / "train" / "parent"
    artifacts_dst = _WORKSPACE / "artifacts"
    if artifacts_src.exists():
        shutil.copytree(artifacts_src, artifacts_dst)
    else:
        artifacts_dst.mkdir(parents=True)

    print(f"[test-merger] model={merger_model}  agent={merger_agent}")

    reflector = AgenticReflector(
        model=reflector_model,
        logger=_Logger(),
        experiment_dir=_FIXTURES,
        merge_threshold=merge_threshold,
        reflector_agent=reflector_agent,
        merger_agent=merger_agent,
        merger_model=merger_model,
    )

    # Force merge directly — no overlap check
    reflector._run_merge(HASH_A, HASH_B, _FIXTURES / "iter")

    merged_files = folder_to_dict(_WORKSPACE_AGENT)
    out_dir = _OUT / "agent_after_merge"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for rel, content in merged_files.items():
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    print(f"\n[test-merger] merged files: {sorted(merged_files.keys())}")
    print(f"[test-merger] saved → {out_dir}/")


if __name__ == "__main__":
    main()
