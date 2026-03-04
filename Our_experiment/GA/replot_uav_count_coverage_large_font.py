import argparse
import json
import os

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redraw the three-group UAV-count coverage plot with larger fonts."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="data/mul_uav/ga_three_group_uav_count_comparison_w4800_g10_i999999.json",
        help="Path to the three-group UAV-count comparison JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/mul_uav/ga_three_group_coverage_vs_uav_w4800_g10_i999999_large_font.png",
        help="Output image path.",
    )
    parser.add_argument("--title-fontsize", type=int, default=30, help="Title font size.")
    parser.add_argument("--axis-fontsize", type=int, default=25, help="Axis label font size.")
    parser.add_argument("--tick-fontsize", type=int, default=21, help="Tick label font size.")
    parser.add_argument("--legend-fontsize", type=int, default=21, help="Legend font size.")
    parser.add_argument("--line-width", type=float, default=3.0, help="Line width.")
    parser.add_argument("--marker-size", type=float, default=10.0, help="Marker size.")
    return parser.parse_args()


def normalize_path(path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.abspath(path))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_plot_series(data):
    results = sorted(data["per_uav_count_results"], key=lambda item: int(item["num_uav"]))
    uav_counts = [int(item["num_uav"]) for item in results]

    ga_off_cov = [float(item["groups"]["ga_offloading"]["metrics"]["mean_coverage_percent"]) for item in results]
    ga_no_cov = [float(item["groups"]["ga_no_offloading"]["metrics"]["mean_coverage_percent"]) for item in results]
    no_ga_cov = [float(item["groups"]["no_ga_offloading"]["metrics"]["mean_coverage_percent"]) for item in results]
    return uav_counts, ga_off_cov, ga_no_cov, no_ga_cov


def build_title(data):
    config = data.get("config", {})
    wind_seed = config.get("wind_seed", "N/A")
    terrain_seed = config.get("terrain_seed", "N/A")
    infra_seed = config.get("infra_seed", "N/A")
    return (
        "Final Coverage Comparison "
    )


def main():
    args = parse_args()
    input_json = normalize_path(args.input_json)
    output_path = normalize_path(args.output_path)

    data = load_json(input_json)
    uav_counts, ga_off_cov, ga_no_cov, no_ga_cov = extract_plot_series(data)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    plt.figure(figsize=(15, 9))
    plt.plot(
        uav_counts,
        ga_off_cov,
        "o-",
        color="blue",
        label="GA + Offloading",
        linewidth=args.line_width,
        markersize=args.marker_size,
    )
    plt.plot(
        uav_counts,
        ga_no_cov,
        "s--",
        color="red",
        label="GA + No Offloading",
        linewidth=args.line_width,
        markersize=args.marker_size,
    )
    plt.plot(
        uav_counts,
        no_ga_cov,
        "^-.",
        color="green",
        label="No-GA + Offloading",
        linewidth=args.line_width,
        markersize=args.marker_size,
    )

    plt.title(build_title(data), fontsize=args.title_fontsize)
    plt.xlabel("Number of UAVs", fontsize=args.axis_fontsize)
    plt.ylabel("Final Coverage (%)", fontsize=args.axis_fontsize)
    plt.xticks(fontsize=args.tick_fontsize)
    plt.yticks(fontsize=args.tick_fontsize)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=args.legend_fontsize, loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=260)
    plt.close()

    print(f"Input JSON: {input_json}")
    print(f"Output image: {output_path}")


if __name__ == "__main__":
    main()
