import argparse
import json
import os


ALGORITHM_ORDER = [
    "ga_deployment_seed_search_2",
    "ga_deployment_seed_search_2_no_offloading",
    "ga_deployment_seed_search_2_rule_based_offloading_2",
]

ALGORITHM_DISPLAY = {
    "ga_deployment_seed_search_2": {
        "variant": "Full (RL+Offloading)",
        "offloading": "Yes",
        "ga": "Yes",
    },
    "ga_deployment_seed_search_2_no_offloading": {
        "variant": "No-Offloading",
        "offloading": "No",
        "ga": "Yes",
    },
    "ga_deployment_seed_search_2_rule_based_offloading_2": {
        "variant": "Simple Greedy + Offloading",
        "offloading": "Yes",
        "ga": "Yes",
    },
}

WIND_ORDER = [11, 23, 4800]
WIND_LABEL = {
    11: "Low Wind",
    23: "Moderate Wind",
    4800: "Strong Wind",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read the GA comparison JSON and print a terminal table for the three algorithms "
            "across Low/Moderate/Strong wind."
        )
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="Our_experiment/GA/data/ga_algorithm_comparison_i999999_g10_winds11_23_4800.json",
        help="Path to the comparison JSON.",
    )
    parser.add_argument(
        "--lifetime-unit",
        type=str,
        default="min",
        choices=["min", "s", "steps"],
        help="Lifetime unit shown in the table.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "plain"],
        help="Table output format.",
    )
    return parser.parse_args()


def resolve_input_json(path):
    candidates = [
        path,
        os.path.join("Our_experiment", "GA", path),
        os.path.join("Our_experiment", "GA", "Our_experiment", "GA", "data", os.path.basename(path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"comparison JSON not found: {path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pm(mean_value, std_value, digits=2):
    return f"{float(mean_value):.{digits}f} ± {float(std_value):.{digits}f}"


def format_lifetime(metrics, unit):
    if unit == "steps":
        return format_pm(metrics["mean_uav_lifetime_steps"], metrics["std_uav_lifetime_steps"], digits=2)
    if unit == "s":
        return format_pm(metrics["mean_uav_lifetime_seconds"], metrics["std_uav_lifetime_seconds"], digits=2)
    return format_pm(
        float(metrics["mean_uav_lifetime_seconds"]) / 60.0,
        float(metrics["std_uav_lifetime_seconds"]) / 60.0,
        digits=2,
    )


def format_uncertainty(metrics):
    return format_pm(metrics["mean_average_uncertainty"], metrics["std_average_uncertainty"], digits=4)


def metric_label(unit):
    if unit == "steps":
        return "Lifetime (steps)"
    if unit == "s":
        return "Lifetime (s)"
    return "Lifetime (min)"


def build_rows(data, lifetime_unit):
    by_algorithm = data.get("comparison_by_algorithm", {})
    rows = []
    for alg_id in ALGORITHM_ORDER:
        alg_data = by_algorithm.get(alg_id)
        if alg_data is None:
            continue

        display = ALGORITHM_DISPLAY.get(
            alg_id,
            {"variant": alg_id, "offloading": "?", "ga": "Yes"},
        )
        row = [
            display["variant"],
            display["offloading"],
            display["ga"],
        ]

        winds = alg_data.get("winds", {})
        for wind_seed in WIND_ORDER:
            wind_metrics = winds.get(str(wind_seed), {}).get("metrics")
            if wind_metrics is None:
                row.extend(["N/A", "N/A"])
            else:
                row.append(format_lifetime(wind_metrics, lifetime_unit))
                row.append(format_uncertainty(wind_metrics))
        rows.append(row)
    return rows


def build_headers(lifetime_unit):
    headers = ["Variant", "Offloading", "GA"]
    for wind_seed in WIND_ORDER:
        headers.append(f"{WIND_LABEL[wind_seed]} {metric_label(lifetime_unit)}")
        headers.append(f"{WIND_LABEL[wind_seed]} Avg Unc")
    return headers


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plain_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main():
    args = parse_args()
    input_json = resolve_input_json(args.input_json)
    data = load_json(input_json)

    headers = build_headers(args.lifetime_unit)
    rows = build_rows(data, args.lifetime_unit)

    print(f"Comparison JSON: {input_json}")
    print("")
    if args.format == "plain":
        print(plain_table(headers, rows))
    else:
        print(markdown_table(headers, rows))


if __name__ == "__main__":
    main()
