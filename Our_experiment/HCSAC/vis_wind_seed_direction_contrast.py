import argparse
import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find wind seeds whose mean wind speed is close to a reference seed "
            "but whose wind direction field is very different, then render the "
            "reference wind field and the selected contrast wind fields."
        )
    )
    parser.add_argument("--reference-wind-seed", type=int, default=4800, help="Reference wind seed.")
    parser.add_argument("--candidate-seed-start", type=int, default=0, help="Candidate wind seed start (inclusive).")
    parser.add_argument("--candidate-seed-end", type=int, default=4999, help="Candidate wind seed end (inclusive).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of contrasting wind seeds to select.")
    parser.add_argument(
        "--subregion-size",
        type=int,
        default=20,
        help="Wind subregion size. Should match the environment grid size.",
    )
    parser.add_argument(
        "--speed-tolerance-ratio",
        type=float,
        default=0.05,
        help="Initial tolerance for mean wind speed similarity, as a fraction of the reference mean speed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Our_experiment/HCSAC/data/wind_direction_contrast",
        help="Output directory for combined figure, individual figures, and report files.",
    )
    parser.add_argument(
        "--combined-figure-path",
        type=str,
        default=None,
        help="Optional custom path for the combined figure.",
    )
    parser.add_argument(
        "--report-json-path",
        type=str,
        default=None,
        help="Optional custom path for the report JSON.",
    )
    parser.add_argument(
        "--report-md-path",
        type=str,
        default=None,
        help="Optional custom path for the Markdown report.",
    )
    return parser.parse_args()


def normalize_path(path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    if path.startswith("Our_experiment/"):
        return os.path.normpath(os.path.join(PROJECT_ROOT, path))
    return os.path.normpath(os.path.abspath(path))


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def find_wind_json_path():
    candidates = [
        os.path.join(PROJECT_ROOT, "OUR_ENV_WITH_WIND_JSON", "wind.json"),
        os.path.join(PROJECT_ROOT, "wind.json"),
        os.path.join(PROJECT_ROOT, "Our_experiment", "HCSAC", "wind.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find wind.json in expected locations.")


def load_wind_matrices(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    width = int(data["width"])
    height = int(data["height"])
    u_matrix = np.array(data["u"]["array"], dtype=np.float64).reshape(height, width)
    v_matrix = np.array(data["v"]["array"], dtype=np.float64).reshape(height, width)
    return u_matrix, v_matrix


def extract_subregion_by_seed(u_matrix, v_matrix, seed, subregion_size):
    height, width = u_matrix.shape
    max_start_x = width - subregion_size
    max_start_y = height - subregion_size
    if max_start_x < 0 or max_start_y < 0:
        raise ValueError(
            f"Original wind data is smaller than the requested subregion size {subregion_size}."
        )

    rng = random.Random(int(seed))
    start_x = rng.randint(0, max_start_x)
    start_y = rng.randint(0, max_start_y)
    u_sub = u_matrix[start_y:start_y + subregion_size, start_x:start_x + subregion_size].copy()
    v_sub = v_matrix[start_y:start_y + subregion_size, start_x:start_x + subregion_size].copy()
    return u_sub, v_sub, int(start_x), int(start_y)


def circular_abs_diff_deg(angle_a_rad, angle_b_rad):
    diff = np.arctan2(np.sin(angle_a_rad - angle_b_rad), np.cos(angle_a_rad - angle_b_rad))
    return float(np.degrees(np.abs(diff)))


def weighted_mean_direction_deg(u_sub, v_sub):
    speed = np.sqrt(u_sub ** 2 + v_sub ** 2)
    total_speed = float(np.sum(speed))
    if total_speed <= 1e-12:
        return 0.0
    mean_u = float(np.sum(u_sub))
    mean_v = float(np.sum(v_sub))
    return float(np.degrees(np.arctan2(mean_v, mean_u)))


def field_direction_difference_deg(ref_u, ref_v, cand_u, cand_v):
    ref_angle = np.arctan2(ref_v, ref_u)
    cand_angle = np.arctan2(cand_v, cand_u)
    diff = np.arctan2(np.sin(cand_angle - ref_angle), np.cos(cand_angle - ref_angle))
    abs_diff = np.abs(diff)
    weights = 0.5 * (np.sqrt(ref_u ** 2 + ref_v ** 2) + np.sqrt(cand_u ** 2 + cand_v ** 2))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return 0.0
    return float(np.degrees(np.sum(abs_diff * weights) / weight_sum))


def summarize_field(seed, u_sub, v_sub, start_x, start_y):
    speed = np.sqrt(u_sub ** 2 + v_sub ** 2)
    mean_speed = float(np.mean(speed))
    std_speed = float(np.std(speed))
    mean_direction_deg = weighted_mean_direction_deg(u_sub, v_sub)
    return {
        "seed": int(seed),
        "start_x": int(start_x),
        "start_y": int(start_y),
        "u": u_sub,
        "v": v_sub,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "mean_direction_deg": mean_direction_deg,
    }


def select_contrasting_seeds(reference, all_candidates, top_k, speed_tolerance_ratio):
    ref_mean_speed = float(reference["mean_speed"])
    tolerance_schedule = [
        float(speed_tolerance_ratio),
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
        float("inf"),
    ]

    unique_schedule = []
    for tol in tolerance_schedule:
        if len(unique_schedule) == 0 or tol != unique_schedule[-1]:
            unique_schedule.append(tol)

    selected = None
    used_tolerance = None
    for tol in unique_schedule:
        subset = [item for item in all_candidates if item["speed_gap_ratio"] <= tol]
        if len(subset) >= top_k:
            subset_sorted = sorted(
                subset,
                key=lambda item: (
                    -item["field_direction_diff_deg"],
                    -item["mean_direction_diff_deg"],
                    item["speed_gap_ratio"],
                ),
            )
            selected = subset_sorted[:top_k]
            used_tolerance = tol
            break

    if selected is None:
        fallback_sorted = sorted(
            all_candidates,
            key=lambda item: (
                item["speed_gap_ratio"],
                -item["field_direction_diff_deg"],
                -item["mean_direction_diff_deg"],
            ),
        )
        selected = fallback_sorted[:top_k]
        used_tolerance = float("inf")

    return selected, used_tolerance


def save_single_wind_map(field, output_path, reference_seed=None):
    title_fs = 22
    axis_label_fs = 18
    tick_fs = 14
    cbar_label_fs = 18
    cbar_tick_fs = 14

    speed = np.sqrt(field["u"] ** 2 + field["v"] ** 2)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    mappable = ax.imshow(
        speed.T,
        cmap="Blues",
        origin="lower",
        interpolation="nearest",
    )

    grid_x = np.arange(field["u"].shape[0])
    grid_y = np.arange(field["u"].shape[1])
    x_mesh, y_mesh = np.meshgrid(grid_x, grid_y, indexing="xy")
    ax.quiver(
        x_mesh,
        y_mesh,
        field["u"].T,
        field["v"].T,
        color="black",
        scale=80,
        width=0.0025,
        alpha=0.65,
    )

    suffix = "" if reference_seed is None else f", ref={reference_seed}"
    ax.set_title(
        (
            f"Wind Seed {field['seed']}{suffix}\n"
            f"mean_speed={field['mean_speed']:.4f}, mean_dir={field['mean_direction_deg']:.1f} deg"
        ),
        fontsize=title_fs,
    )
    ax.set_xlabel("Grid X", fontsize=axis_label_fs)
    ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Wind Speed", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_combined_wind_map(reference, selected_fields, output_path):
    title_fs = 20
    axis_label_fs = 16
    tick_fs = 12
    suptitle_fs = 24
    cbar_label_fs = 18
    cbar_tick_fs = 14

    all_fields = [reference] + selected_fields
    vmax = max(float(np.max(np.sqrt(item["u"] ** 2 + item["v"] ** 2))) for item in all_fields)
    vmin = 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    mappable = None
    for idx, field in enumerate(all_fields):
        ax = axes[idx]
        speed = np.sqrt(field["u"] ** 2 + field["v"] ** 2)
        mappable = ax.imshow(
            speed.T,
            cmap="Blues",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        grid_x = np.arange(field["u"].shape[0])
        grid_y = np.arange(field["u"].shape[1])
        x_mesh, y_mesh = np.meshgrid(grid_x, grid_y, indexing="xy")
        ax.quiver(
            x_mesh,
            y_mesh,
            field["u"].T,
            field["v"].T,
            color="black",
            scale=80,
            width=0.0022,
            alpha=0.65,
        )

        if idx == 0:
            title = (
                f"Reference w={field['seed']}\n"
                f"mean_speed={field['mean_speed']:.4f}, mean_dir={field['mean_direction_deg']:.1f} deg"
            )
        else:
            title = (
                f"Seed {field['seed']}\n"
                f"mean_speed={field['mean_speed']:.4f}, field_dir_diff={field['field_direction_diff_deg']:.1f} deg"
            )
        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("Grid X", fontsize=axis_label_fs)
        ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

    for idx in range(len(all_fields), len(axes)):
        axes[idx].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Wind Speed", fontsize=cbar_label_fs, labelpad=12)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.suptitle(
        (
            f"Wind Fields Similar in Mean Speed to w={reference['seed']}, "
            f"but Strongly Different in Direction"
        ),
        fontsize=suptitle_fs,
        x=0.49,
        y=0.98,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def build_markdown_report(reference, selected_fields, report_json_path, combined_figure_path, individual_dir, used_tolerance, candidate_seed_start, candidate_seed_end):
    lines = []
    lines.append("# Wind Seed Direction Contrast Report")
    lines.append("")
    lines.append(f"- Reference wind seed: `{reference['seed']}`")
    lines.append(f"- Candidate seed range: `{candidate_seed_start}-{candidate_seed_end}`")
    lines.append(f"- Reference mean wind speed: `{reference['mean_speed']:.6f}`")
    lines.append(f"- Reference mean direction: `{reference['mean_direction_deg']:.3f}` deg")
    lines.append(f"- Applied mean-speed tolerance ratio: `{used_tolerance}`")
    lines.append(f"- Combined figure: `{combined_figure_path}`")
    lines.append(f"- JSON report: `{report_json_path}`")
    lines.append(f"- Individual figure directory: `{individual_dir}`")
    lines.append("")
    lines.append("## Selected Contrast Seeds")
    lines.append("")
    for item in selected_fields:
        lines.append(
            "- "
            f"seed `{item['seed']}`: mean_speed=`{item['mean_speed']:.6f}`, "
            f"speed_gap_ratio=`{item['speed_gap_ratio']:.6f}`, "
            f"mean_direction=`{item['mean_direction_deg']:.3f}` deg, "
            f"mean_direction_diff=`{item['mean_direction_diff_deg']:.3f}` deg, "
            f"field_direction_diff=`{item['field_direction_diff_deg']:.3f}` deg, "
            f"start=(`{item['start_x']}`, `{item['start_y']}`)"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")
    if args.candidate_seed_end < args.candidate_seed_start:
        raise ValueError("candidate-seed-end must be >= candidate-seed-start")
    if args.reference_wind_seed < args.candidate_seed_start or args.reference_wind_seed > args.candidate_seed_end:
        pass

    wind_json_path = find_wind_json_path()
    full_u, full_v = load_wind_matrices(wind_json_path)

    ref_u, ref_v, ref_x, ref_y = extract_subregion_by_seed(
        full_u,
        full_v,
        seed=args.reference_wind_seed,
        subregion_size=args.subregion_size,
    )
    reference = summarize_field(args.reference_wind_seed, ref_u, ref_v, ref_x, ref_y)

    all_candidates = []
    for seed in range(int(args.candidate_seed_start), int(args.candidate_seed_end) + 1):
        if int(seed) == int(args.reference_wind_seed):
            continue
        cand_u, cand_v, cand_x, cand_y = extract_subregion_by_seed(
            full_u,
            full_v,
            seed=seed,
            subregion_size=args.subregion_size,
        )
        candidate = summarize_field(seed, cand_u, cand_v, cand_x, cand_y)
        speed_gap_ratio = abs(candidate["mean_speed"] - reference["mean_speed"]) / max(reference["mean_speed"], 1e-12)
        mean_direction_diff_deg = circular_abs_diff_deg(
            math.radians(candidate["mean_direction_deg"]),
            math.radians(reference["mean_direction_deg"]),
        )
        field_direction_diff_deg = field_direction_difference_deg(
            reference["u"],
            reference["v"],
            candidate["u"],
            candidate["v"],
        )
        candidate["speed_gap_ratio"] = float(speed_gap_ratio)
        candidate["mean_direction_diff_deg"] = float(mean_direction_diff_deg)
        candidate["field_direction_diff_deg"] = float(field_direction_diff_deg)
        all_candidates.append(candidate)

    selected_fields, used_tolerance = select_contrasting_seeds(
        reference=reference,
        all_candidates=all_candidates,
        top_k=int(args.top_k),
        speed_tolerance_ratio=float(args.speed_tolerance_ratio),
    )

    output_dir = normalize_path(args.output_dir)
    run_dir = os.path.join(
        output_dir,
        f"ref_w{int(args.reference_wind_seed)}_top{int(args.top_k)}",
    )
    os.makedirs(run_dir, exist_ok=True)

    combined_figure_path = (
        normalize_path(args.combined_figure_path)
        if args.combined_figure_path
        else os.path.join(run_dir, f"wind_direction_contrast_ref{int(args.reference_wind_seed)}.png")
    )
    report_json_path = (
        normalize_path(args.report_json_path)
        if args.report_json_path
        else os.path.join(run_dir, f"wind_direction_contrast_ref{int(args.reference_wind_seed)}.json")
    )
    report_md_path = (
        normalize_path(args.report_md_path)
        if args.report_md_path
        else os.path.join(run_dir, f"wind_direction_contrast_ref{int(args.reference_wind_seed)}.md")
    )
    individual_dir = os.path.join(run_dir, "individual_wind_fields")
    os.makedirs(individual_dir, exist_ok=True)

    save_single_wind_map(
        reference,
        os.path.join(individual_dir, f"wind_seed_{int(reference['seed'])}.png"),
        reference_seed=int(reference["seed"]),
    )
    for item in selected_fields:
        save_single_wind_map(
            item,
            os.path.join(individual_dir, f"wind_seed_{int(item['seed'])}.png"),
            reference_seed=int(reference["seed"]),
        )

    save_combined_wind_map(reference, selected_fields, combined_figure_path)

    payload = {
        "reference_seed": int(reference["seed"]),
        "candidate_seed_range": {
            "start": int(args.candidate_seed_start),
            "end": int(args.candidate_seed_end),
        },
        "subregion_size": int(args.subregion_size),
        "wind_json_path": wind_json_path,
        "selection": {
            "top_k": int(args.top_k),
            "requested_speed_tolerance_ratio": float(args.speed_tolerance_ratio),
            "used_speed_tolerance_ratio": (
                None if not np.isfinite(used_tolerance) else float(used_tolerance)
            ),
            "coverage_note": "Selected by similar mean wind speed and large field-wise direction difference.",
        },
        "reference_field": {
            "seed": int(reference["seed"]),
            "start_x": int(reference["start_x"]),
            "start_y": int(reference["start_y"]),
            "mean_speed": float(reference["mean_speed"]),
            "std_speed": float(reference["std_speed"]),
            "mean_direction_deg": float(reference["mean_direction_deg"]),
        },
        "selected_fields": [
            {
                "seed": int(item["seed"]),
                "start_x": int(item["start_x"]),
                "start_y": int(item["start_y"]),
                "mean_speed": float(item["mean_speed"]),
                "std_speed": float(item["std_speed"]),
                "speed_gap_ratio": float(item["speed_gap_ratio"]),
                "mean_direction_deg": float(item["mean_direction_deg"]),
                "mean_direction_diff_deg": float(item["mean_direction_diff_deg"]),
                "field_direction_diff_deg": float(item["field_direction_diff_deg"]),
            }
            for item in selected_fields
        ],
        "outputs": {
            "combined_figure": combined_figure_path,
            "markdown_report": report_md_path,
            "json_report": report_json_path,
            "individual_dir": individual_dir,
        },
    }

    ensure_parent_dir(report_json_path)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    report_md = build_markdown_report(
        reference=reference,
        selected_fields=selected_fields,
        report_json_path=report_json_path,
        combined_figure_path=combined_figure_path,
        individual_dir=individual_dir,
        used_tolerance=used_tolerance,
        candidate_seed_start=int(args.candidate_seed_start),
        candidate_seed_end=int(args.candidate_seed_end),
    )
    ensure_parent_dir(report_md_path)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("-" * 80)
    print("Wind direction contrast analysis completed.")
    print(f"Reference wind seed: {reference['seed']}")
    print(f"Reference mean wind speed: {reference['mean_speed']:.6f}")
    print(f"Reference mean direction: {reference['mean_direction_deg']:.3f} deg")
    print(f"Used speed tolerance ratio: {used_tolerance}")
    print("Selected contrast seeds:")
    for item in selected_fields:
        print(
            f"  seed={item['seed']}, mean_speed={item['mean_speed']:.6f}, "
            f"speed_gap_ratio={item['speed_gap_ratio']:.6f}, "
            f"mean_direction_diff={item['mean_direction_diff_deg']:.3f} deg, "
            f"field_direction_diff={item['field_direction_diff_deg']:.3f} deg"
        )
    print(f"Combined figure: {combined_figure_path}")
    print(f"Individual figures dir: {individual_dir}")
    print(f"JSON report: {report_json_path}")
    print(f"Markdown report: {report_md_path}")


if __name__ == "__main__":
    main()
