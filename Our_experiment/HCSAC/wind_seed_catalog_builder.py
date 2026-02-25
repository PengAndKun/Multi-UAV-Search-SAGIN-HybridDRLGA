import os
import sys
import json
import csv
import argparse
import random

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Our_experiment.HCSAC.wind import extract_wind_subregion, wind_speed_mean


CLASS_LABELS = ["Low Wind", "Moderate Wind", "Strong Wind"]


def find_wind_json_path():
    candidates = [
        os.path.join(project_root, "OUR_ENV_WITH_WIND_JSON", "wind.json"),
        os.path.join(project_root, "wind.json"),
        os.path.join(project_root, "Our_experiment", "HCSAC", "wind.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find wind.json in expected locations.")


def classify_tertiles(values):
    q1 = float(np.quantile(values, 1.0 / 3.0))
    q2 = float(np.quantile(values, 2.0 / 3.0))

    classes = []
    for v in values:
        if v <= q1:
            classes.append(CLASS_LABELS[0])
        elif v <= q2:
            classes.append(CLASS_LABELS[1])
        else:
            classes.append(CLASS_LABELS[2])
    return classes, q1, q2


def representative_seed(rows, class_name):
    class_rows = [r for r in rows if r["class"] == class_name]
    class_rows_sorted = sorted(class_rows, key=lambda x: x["wind_speed_mean"])
    return class_rows_sorted[len(class_rows_sorted) // 2]["seed"]


def build_catalog(
    num_seeds,
    subregion_size,
    csv_path,
    json_path,
    report_path,
    terrain_seed_mode,
    fixed_terrain_seed,
    terrain_grid_size,
):
    wind_json_path = find_wind_json_path()

    rows = []
    means = []
    terrain_means = []

    for seed in range(num_seeds):
        random.seed(seed)
        np.random.seed(seed)
        u_sub, v_sub, start_x, start_y = extract_wind_subregion(wind_json_path, subregion_size=subregion_size)
        mean_speed = float(wind_speed_mean(u_sub, v_sub))
        terrain_seed = seed if terrain_seed_mode == "match_wind" else fixed_terrain_seed
        terrain_rng = np.random.default_rng(terrain_seed)
        terrain_matrix = terrain_rng.integers(1, 5, size=(terrain_grid_size, terrain_grid_size))
        terrain_mean = float(np.mean(terrain_matrix))
        means.append(mean_speed)
        terrain_means.append(terrain_mean)
        rows.append(
            {
                "seed": seed,
                "wind_speed_mean": mean_speed,
                "terrain_seed": int(terrain_seed),
                "terrain_difficulty_mean": terrain_mean,
                "start_x": int(start_x),
                "start_y": int(start_y),
            }
        )

    classes, q1, q2 = classify_tertiles(np.array(means, dtype=np.float64))
    for row, cls in zip(rows, classes):
        row["class"] = cls

    reps = {
        "Low Wind": representative_seed(rows, "Low Wind"),
        "Moderate Wind": representative_seed(rows, "Moderate Wind"),
        "Strong Wind": representative_seed(rows, "Strong Wind"),
    }

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "wind_speed_mean",
                "terrain_seed",
                "terrain_difficulty_mean",
                "start_x",
                "start_y",
                "class",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    class_to_seeds = {label: [] for label in CLASS_LABELS}
    for row in rows:
        class_to_seeds[row["class"]].append(int(row["seed"]))

    payload = {
        "num_seeds": num_seeds,
        "subregion_size": subregion_size,
        "wind_json_path": wind_json_path,
        "terrain_seed_mode": terrain_seed_mode,
        "fixed_terrain_seed": fixed_terrain_seed if terrain_seed_mode == "fixed" else None,
        "terrain_grid_size": terrain_grid_size,
        "thresholds": {"q1": q1, "q2": q2},
        "representative_seeds": reps,
        "class_to_seeds": class_to_seeds,
        "summary": {
            "global_mean": float(np.mean(means)),
            "global_std": float(np.std(means)),
            "min_mean_speed": float(np.min(means)),
            "max_mean_speed": float(np.max(means)),
            "terrain_difficulty_mean_global": float(np.mean(terrain_means)),
            "terrain_difficulty_std_global": float(np.std(terrain_means)),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Wind Seed Catalog Report")
    lines.append("")
    lines.append(f"- Total seeds analyzed: `{num_seeds}`")
    lines.append(f"- Wind JSON path: `{wind_json_path}`")
    lines.append(f"- Subregion size: `{subregion_size}`")
    lines.append(f"- Terrain seed mode: `{terrain_seed_mode}`")
    if terrain_seed_mode == "fixed":
        lines.append(f"- Fixed terrain seed: `{fixed_terrain_seed}`")
    lines.append(f"- Terrain grid size: `{terrain_grid_size}`")
    lines.append(f"- Threshold q1 (Low/Moderate): `{q1:.6f}`")
    lines.append(f"- Threshold q2 (Moderate/Strong): `{q2:.6f}`")
    lines.append("")
    lines.append("## Representative Seeds")
    lines.append("")
    for label in CLASS_LABELS:
        lines.append(f"- {label}: `{reps[label]}`")
    lines.append("")
    lines.append("## Class Counts")
    lines.append("")
    for label in CLASS_LABELS:
        lines.append(f"- {label}: `{len(class_to_seeds[label])}`")
    lines.append("")
    lines.append("## Global Stats")
    lines.append("")
    lines.append(f"- Mean wind speed: `{payload['summary']['global_mean']:.6f}`")
    lines.append(f"- Std wind speed: `{payload['summary']['global_std']:.6f}`")
    lines.append(f"- Min mean speed: `{payload['summary']['min_mean_speed']:.6f}`")
    lines.append(f"- Max mean speed: `{payload['summary']['max_mean_speed']:.6f}`")
    lines.append(f"- Terrain mean difficulty: `{payload['summary']['terrain_difficulty_mean_global']:.6f}`")
    lines.append(f"- Terrain difficulty std: `{payload['summary']['terrain_difficulty_std_global']:.6f}`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Catalog CSV saved: {csv_path}")
    print(f"Catalog JSON saved: {json_path}")
    print(f"Catalog report saved: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build wind seed catalog and split seeds into 3 wind classes.")
    parser.add_argument("--num-seeds", type=int, default=5000, help="Number of wind seeds to analyze.")
    parser.add_argument("--subregion-size", type=int, default=20, help="Wind subregion size.")
    parser.add_argument(
        "--terrain-seed-mode",
        type=str,
        default="match_wind",
        choices=["match_wind", "fixed"],
        help="How terrain seed is assigned for catalog records.",
    )
    parser.add_argument(
        "--fixed-terrain-seed",
        type=int,
        default=0,
        help="Terrain seed used when --terrain-seed-mode=fixed.",
    )
    parser.add_argument(
        "--terrain-grid-size",
        type=int,
        default=20,
        help="Grid size used to summarize terrain difficulty seed effect.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="Our_experiment/HCSAC/data/wind_seed_catalog_5000.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="Our_experiment/HCSAC/data/wind_seed_classes_5000.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="Our_experiment/HCSAC/data/wind_seed_classes_report_5000.md",
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_catalog(
        num_seeds=args.num_seeds,
        subregion_size=args.subregion_size,
        csv_path=args.csv_path,
        json_path=args.json_path,
        report_path=args.report_path,
        terrain_seed_mode=args.terrain_seed_mode,
        fixed_terrain_seed=args.fixed_terrain_seed,
        terrain_grid_size=args.terrain_grid_size,
    )


if __name__ == "__main__":
    main()
