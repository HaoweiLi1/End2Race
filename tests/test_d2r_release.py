#!/usr/bin/env python3
"""Schema tests for D2R registry rows and locked release constants."""

import hashlib
import tempfile
from pathlib import Path

from d2r import EVIDENCE_RELPATH, REGISTRY_OPENED_AT
from d2r.data import make_registry_rows
from d2r.release import _verify_manifest


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def main():
    episodes = []
    for index in range(1928):
        token = f"{index:064x}"
        episodes.append(
            {
                "l2_id": f"L2:{token}",
                "l3_id": f"L3:{token}",
                "l4_id": f"L4:{index // 12:064x}",
                "map_name": "Austin" if index % 2 == 0 else "Hockenheim",
            }
        )
    rows = make_registry_rows(episodes)
    check("row-count", len(rows) == 1928)
    check("row-id-unique", len({row["row_id"] for row in rows}) == 1928)
    check("stage", all(row["stage"] == "D2R-G" for row in rows))
    check("use", all(row["use_class"] == "probe_fit" for row in rows))
    check("decision", all(row["decision_effect"] == "representation_choice" for row in rows))
    check("final-pool", all(row["final_pool"] == "false" for row in rows))
    check("opened-at", all(row["opened_at_utc"] == REGISTRY_OPENED_AT for row in rows))
    check("evidence", all(row["evidence_relpath"] == EVIDENCE_RELPATH for row in rows))

    with tempfile.TemporaryDirectory(prefix="d2r_release_test_") as temporary:
        root = Path(temporary)
        payload = root / "payload.txt"
        payload.write_text("locked\n", encoding="utf-8")
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        (root / "output_manifest.sha256").write_text(
            f"{digest}  payload.txt\n", encoding="utf-8"
        )
        _verify_manifest(root)
        payload.write_text("corrupt\n", encoding="utf-8")
        try:
            _verify_manifest(root)
        except ValueError as error:
            check("manifest-corruption-message", "hash mismatch" in str(error))
        else:
            raise AssertionError("FAIL manifest-corruption-undetected")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
