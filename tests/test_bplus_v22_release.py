#!/usr/bin/env python3
"""Atomic source-preflight release and corruption regression."""

import json
from pathlib import Path
import tempfile

from bplus_v22 import OWNER_DECISION
from bplus_v22.release import create_source_preflight, validate_source_preflight


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        release = Path(temporary) / "preflight"
        result = create_source_preflight(
            release, "2026-07-11T22:30:00+08:00", repo_root="."
        )
        assert result["passed"] and result["source_files"] >= 20
        validation = validate_source_preflight(release, repo_root=".")
        assert validation["passed"] and validation["violations"] == []
        authority = json.loads((release / "authority.json").read_text())
        assert authority["owner_decision"] == OWNER_DECISION
        assert authority["old_d2r_gate_passed"] is False
        try:
            create_source_preflight(
                release, "2026-07-11T22:31:00+08:00", repo_root="."
            )
            raise AssertionError("source preflight overwrite accepted")
        except FileExistsError:
            pass
        environment = release / "environment.json"
        environment.write_text(environment.read_text() + "corruption\n")
        corrupt = validate_source_preflight(release, repo_root=".")
        assert not corrupt["passed"]
        assert any("hash mismatch" in item for item in corrupt["violations"])
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
