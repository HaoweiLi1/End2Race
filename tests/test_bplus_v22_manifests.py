#!/usr/bin/env python3
"""Task-8 outcome-scoped panel and Cartesian-job contracts."""

from bplus_v22.manifests import (
    DEVELOPMENT_PANELS,
    PANEL_REPRESENTATIVE,
    build_manifests,
)


def main() -> None:
    first = build_manifests(".")
    second = build_manifests(".")
    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "development": 288,
        "training": 1640,
        "witness": 67,
        "recoverability": 0,
        "smoke": 8,
    }
    development = first["development"]
    assert [int(row["manifest_order"]) for row in development] == list(range(288))
    assert len({row["l2_id"] for row in development}) == 288
    assert {row["panel"] for row in development} == set(DEVELOPMENT_PANELS)
    assert all(
        sum(row["panel"] == panel for row in development) == 96
        for panel in DEVELOPMENT_PANELS
    )
    assert all(
        (row["panel"] == PANEL_REPRESENTATIVE)
        == (row["estimate_class"] == "distributional_development")
        for row in development
    )
    assert not ({row["l2_id"] for row in development} & {row["l2_id"] for row in first["training"]})
    assert all(row["held_out_policy_generalization"] == "false" for row in first["witness"])
    assert {row["selection_stratum"] for row in first["smoke"]} == {
        f"{map_name}:{skill}"
        for map_name in ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
        for skill in ("skill_F", "skill_S")
    }
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
