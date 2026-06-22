"""Test reflector (tweak) using fixtures copied from experiment13.

Candidate A (seed) is loaded from fixtures/candidates/ into workspace/agent/.
Artifacts come from fixtures/iter/train/parent/ (one real parent run).

WARNING: overwrites workspace/ during the run (same as production).
"""

import json
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


class _Logger:
    def log_reflector(self, input_tokens, output_tokens, message, raw, cost=0.0, steps=None):
        print(f"[test-reflector] tokens in={input_tokens} out={output_tokens} "
              f"cost=${cost:.4f} steps={steps}")
    def log_reflector_changes(self, changes):
        print(f"[test-reflector] changes ({len(changes)} chars):")
        print(changes[:600] + ("..." if len(changes) > 600 else ""))


def main() -> None:
    cfg             = yaml.safe_load((_ROOT / "config.yaml").read_text())
    reflector_model = cfg["reflector"]["md"]["model"]
    reflector_agent = cfg["reflector"].get("agent", "reflector")
    merger_model    = cfg["merger"]["md"].get("model", reflector_model)
    merger_agent    = cfg["merger"].get("agent", "merger")
    merge_threshold = cfg["merger"].get("merge_threshold", 0.5)

    # Point candidate_store at our local fixtures
    candidate_store.configure(_FIXTURES / "candidates")

    # Set up workspace/agent/ from local fixture candidate
    print(f"[test-reflector] restoring {HASH_A[:12]}… → workspace/agent/")
    if _WORKSPACE.exists():
        shutil.rmtree(_WORKSPACE)
    _WORKSPACE.mkdir(parents=True)
    candidate_store.restore_workspace(HASH_A, _WORKSPACE_AGENT)

    print(f"[test-reflector] agent files: {sorted(folder_to_dict(_WORKSPACE_AGENT).keys())}")
    print(f"[test-reflector] model={reflector_model}  agent={reflector_agent}")

    reflector = AgenticReflector(
        model=reflector_model,
        logger=_Logger(),
        experiment_dir=_FIXTURES,   # iter_dir = fixtures/iter/
        merge_threshold=merge_threshold,
        reflector_agent=reflector_agent,
        merger_agent=merger_agent,
        merger_model=merger_model,
    )

    # iter_dir is fixtures/iter/ — _copy_artifacts will read fixtures/iter/train/parent/
    reflector._run_tweak(_FIXTURES / "iter")

    new_files = folder_to_dict(_WORKSPACE_AGENT)
    out_dir = _OUT / "agent_after_tweak"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for rel, content in new_files.items():
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    print(f"\n[test-reflector] output files: {sorted(new_files.keys())}")
    print(f"[test-reflector] saved → {out_dir}/")


if __name__ == "__main__":
    main()
