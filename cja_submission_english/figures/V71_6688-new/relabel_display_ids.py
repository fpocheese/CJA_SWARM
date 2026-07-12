#!/usr/bin/env python3
"""Relabel offensive-aircraft display IDs without changing simulation data."""

import json
from pathlib import Path

import numpy as np


OFFENSIVE_TRAJECTORY_KEYS = {
    "off_x", "off_y", "off_z", "off_v", "off_heading", "off_gamma",
    "off_an_pitch", "off_an_yaw", "off_lbc", "off_alive", "off_hit",
    "off_d_hvt",
}
TARGET_KEYS = {
    "def_initial_target", "def_assigned_target", "def_current_attack_target",
    "def_ltgt",
}
OFFENSIVE_GAME_KEYS = {
    "decoy_role_decoy", "decoy_role_pen", "decoy_role_stealth",
    "decoy_lock_pressure", "pen_P_pen", "esc_E_esc", "hvt_P_hit",
    "hvt_rho", "hvt_closing",
}
OFFENSIVE_DEFENSIVE_MATRIX_KEYS = {"esc_Gamma_matrix", "esc_Xi_matrix"}


CASES = {
    "6v6": {
        "source": Path("hil_offense_current_target_rerun_20260620/6v6"),
        "old_to_new": [0, 3, 5, 1, 2, 4],
    },
    "8v8": {
        "source": Path("hil_offense_current_target_rerun_20260620/8v8"),
        "old_to_new": [0, 5, 6, 7, 3, 2, 4, 1],
    },
    "10v10": {
        "source": Path("10v10_success_20260620/20260620_171139_10v10/seed100013"),
        "old_to_new": [0, 3, 5, 7, 9, 1, 6, 8, 2, 4],
    },
}


def remap_targets(values: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    valid = (result >= 0) & (result < len(old_to_new))
    result[valid] = old_to_new[result[valid].astype(int)]
    return result


def transform_trajectory(source: Path, destination: Path, old_to_new: np.ndarray) -> None:
    inverse = np.argsort(old_to_new)
    raw = np.load(source / "trajectory_data.npz", allow_pickle=True)
    transformed = {}
    for key in raw.files:
        values = raw[key]
        if key in OFFENSIVE_TRAJECTORY_KEYS:
            values = values[inverse]
        elif key == "actor_actions":
            values = values[:, inverse, ...]
        elif key == "assign_cost":
            values = values[:, :, inverse]
        elif key in TARGET_KEYS:
            values = remap_targets(values, old_to_new)
        transformed[key] = values
    np.savez_compressed(destination / "trajectory_data.npz", **transformed)


def transform_game(source: Path, destination: Path, old_to_new: np.ndarray) -> None:
    inverse = np.argsort(old_to_new)
    raw = np.load(source / "game_data.npz", allow_pickle=True)
    transformed = {}
    for key in raw.files:
        values = raw[key]
        if key in OFFENSIVE_GAME_KEYS:
            values = values[inverse]
        elif key in OFFENSIVE_DEFENSIVE_MATRIX_KEYS:
            values = values[:, inverse, :]
        elif key in TARGET_KEYS:
            values = remap_targets(values, old_to_new)
        transformed[key] = values
    np.savez_compressed(destination / "game_data.npz", **transformed)


def transform_summary(source: Path, destination: Path, old_to_new: np.ndarray) -> None:
    inverse = np.argsort(old_to_new)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))

    for key in ("hit_indices",):
        if key in summary:
            summary[key] = [int(old_to_new[int(i)]) for i in summary[key]]
    for key in ("best_agent", "hitter"):
        if key in summary and summary[key] is not None:
            summary[key] = int(old_to_new[int(summary[key])])
    for key in ("hit_step", "death_step"):
        if key in summary:
            summary[key] = {
                str(int(old_to_new[int(old_id)])): value
                for old_id, value in summary[key].items()
            }
    if "min_dist_per_agent_m" in summary:
        old_values = summary["min_dist_per_agent_m"]
        summary["min_dist_per_agent_m"] = [old_values[int(i)] for i in inverse]
    if "clone_map" in summary:
        old_clone_map = summary["clone_map"]
        summary["clone_map"] = {
            str(new_id): old_clone_map[str(int(inverse[new_id]))]
            for new_id in range(len(inverse))
        }

    summary["display_id_relabel"] = {
        "purpose": "Display-only offensive-aircraft ID permutation",
        "old_to_new": {str(i): int(v) for i, v in enumerate(old_to_new)},
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate(source: Path, destination: Path, old_to_new: np.ndarray) -> None:
    inverse = np.argsort(old_to_new)
    old = np.load(source / "trajectory_data.npz", allow_pickle=True)
    new = np.load(destination / "trajectory_data.npz", allow_pickle=True)
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))

    assert np.array_equal(new["off_x"], old["off_x"][inverse])
    assert np.array_equal(new["def_x"], old["def_x"])
    assert np.array_equal(new["assign_cost"], old["assign_cost"][:, :, inverse])
    assert np.array_equal(
        new["def_current_attack_target"],
        remap_targets(old["def_current_attack_target"], old_to_new),
    )
    assert np.array_equal(new["def_current_attack_target"], new["def_ltgt"])
    assert summary["hit_indices"] == [int(old_to_new[4])]


def main() -> None:
    output_root = Path("display_relabel_20260620")
    for case, config in CASES.items():
        source = config["source"]
        old_to_new = np.asarray(config["old_to_new"], dtype=int)
        destination = output_root / case
        destination.mkdir(parents=True, exist_ok=True)
        transform_trajectory(source, destination, old_to_new)
        transform_game(source, destination, old_to_new)
        transform_summary(source, destination, old_to_new)
        validate(source, destination, old_to_new)
        print(f"{case}: A4 -> A{old_to_new[4]} ({destination})")


if __name__ == "__main__":
    main()
