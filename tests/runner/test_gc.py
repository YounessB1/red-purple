"""Unit test for the orphan-workdir sweeper — no server/LLM/Docker needed.

Run directly:  python3 tests/runner/test_gc.py
"""

import os
import tempfile
import time
from pathlib import Path

from source.agent.runner import _WORKDIR_PREFIX, sweep_orphan_workdirs

GRACE = 3600  # 60 min, the server default


def _make_workdir(base: Path, run_id: str, *, age_seconds: float) -> Path:
    """Create a fake `agent_<run_id>_<suffix>/` with a file, backdated by age."""
    d = Path(tempfile.mkdtemp(prefix=f"{_WORKDIR_PREFIX}{run_id}_", dir=base))
    (d / "opencode.db").write_text("x", encoding="utf-8")
    past = time.time() - age_seconds
    os.utime(d, (past, past))
    return d


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # 1. active run (in registry) — must survive even though it's old.
        active = _make_workdir(base, "run-aaaa0001", age_seconds=GRACE + 600)
        # 2. orphan, old (beyond grace) — must be deleted.
        old_orphan = _make_workdir(base, "run-bbbb0002", age_seconds=GRACE + 600)
        # 3. orphan, fresh (within grace) — must survive.
        fresh_orphan = _make_workdir(base, "run-cccc0003", age_seconds=60)
        # 4. unrelated dir (wrong prefix) — must be ignored entirely.
        other = base / "not-an-agent-dir"
        other.mkdir()
        os.utime(other, (time.time() - GRACE - 600, time.time() - GRACE - 600))

        stats = sweep_orphan_workdirs({"run-aaaa0001"}, GRACE, base_dir=base)

        assert active.exists(), "active run's workdir was wrongly deleted"
        assert not old_orphan.exists(), "old orphan was not deleted"
        assert fresh_orphan.exists(), "fresh orphan deleted before grace elapsed"
        assert other.exists(), "non-workdir directory was touched"
        assert stats["removed"] == 1, f"expected 1 removed, got {stats}"
        assert stats["freed_bytes"] >= 1, f"freed_bytes not counted: {stats}"
        assert stats["kept"] == 2, f"expected 2 kept (active + fresh), got {stats}"

        # Second sweep is a no-op (nothing left to remove).
        stats2 = sweep_orphan_workdirs({"run-aaaa0001"}, GRACE, base_dir=base)
        assert stats2["removed"] == 0, f"second sweep should remove nothing: {stats2}"

    print("test_gc: OK")


if __name__ == "__main__":
    main()
