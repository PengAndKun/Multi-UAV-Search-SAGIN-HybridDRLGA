import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.HCSAC.wind import extract_wind_subregion


DEFAULT_CATALOG_JSON = (
    "Our_experiment/HCSAC/Our_experiment/HCSAC/data/wind_seed_classes_5000.json"
)
DEFAULT_OUTPUT_PATH = (
    "Our_experiment/HCSAC/Our_experiment/HCSAC/data/"
    "wind_field_catalog_w{wind_seed}.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read wind metadata from wind_seed_classes_5000.json and draw a specified wind field "
            "with larger fonts and sparser/thicker arrows."
        )
    )
    parser.add_argument(
        "--wind-catalog-json",
        type=str,
        default=DEFAULT_CATALOG_JSON,
        help="Path to wind seed catalog JSON.",
    )
    parser.add_argument(
        "--wind-seed",
        type=int,
        default=None,
        help="Specific wind seed to visualize. If omitted, use the representative seed of --wind-class.",
    )
    parser.add_argument(
        "--wind-class",
        type=str,
        default="Strong Wind",
        choices=["Low Wind", "Moderate Wind", "Strong Wind"],
        help="Wind class used when --wind-seed is not specified.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path. Supports {wind_seed} placeholder.",
    )
    parser.add_argument("--fig-width", type=float, default=11.0, help="Figure width.")
    parser.add_argument("--fig-height", type=float, default=9.0, help="Figure height.")
    parser.add_argument("--title-fontsize", type=int, default=30, help="Title font size.")
    parser.add_argument("--axis-fontsize", type=int, default=24, help="Axis label font size.")
    parser.add_argument("--tick-fontsize", type=int, default=19, help="Tick font size.")
    parser.add_argument("--cbar-label-fontsize", type=int, default=22, help="Colorbar label font size.")
    parser.add_argument("--cbar-tick-fontsize", type=int, default=17, help="Colorbar tick font size.")
    parser.add_argument(
        "--arrow-step",
        type=int,
        default=2,
        help="Sample every N grid cells for quiver arrows.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=42.0,
        help="Quiver scale. Smaller values make arrows longer.",
    )
    parser.add_argument(
        "--arrow-width",
        type=float,
        default=0.0055,
        help="Quiver arrow shaft width.",
    )
    parser.add_argument("--arrow-alpha", type=float, default=0.9, help="Quiver alpha.")
    parser.add_argument("--arrow-headwidth", type=float, default=4.6, help="Quiver head width.")
    parser.add_argument("--arrow-headlength", type=float, default=6.2, help="Quiver head length.")
    parser.add_argument("--arrow-headaxislength", type=float, default=5.6, help="Quiver head axis length.")
    parser.add_argument("--dpi", type=int, default=260, help="Output DPI.")
    return parser.parse_args()


def normalize_path(path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.abspath(os.path.join(PROJECT_ROOT, path)))


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_wind_seed(catalog, wind_seed, wind_class):
    if wind_seed is not None:
        return int(wind_seed)
    representatives = catalog.get("representative_seeds", {})
    if wind_class not in representatives:
        raise KeyError(f"wind class not found in representative_seeds: {wind_class}")
    return int(representatives[wind_class])


def load_wind_field(catalog, wind_seed):
    wind_json_path = catalog.get("wind_json_path")
    if not wind_json_path:
        raise KeyError("wind_json_path not found in catalog JSON.")
    subregion_size = int(catalog.get("subregion_size", 20))
    random.seed(int(wind_seed))
    wind_u, wind_v, start_x, start_y = extract_wind_subregion(wind_json_path, subregion_size=subregion_size)
    return wind_u, wind_v, start_x, start_y


def save_wind_field_map(
    wind_u,
    wind_v,
    output_path,
    wind_seed,
    args,
):
    speed = np.sqrt(wind_u ** 2 + wind_v ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(args.fig_width, args.fig_height))
    mappable = ax.imshow(
        speed.T,
        cmap="Blues",
        origin="lower",
        interpolation="nearest",
    )

    step = max(1, int(args.arrow_step))
    grid_x = np.arange(0, wind_u.shape[0], step)
    grid_y = np.arange(0, wind_u.shape[1], step)
    x_mesh, y_mesh = np.meshgrid(grid_x, grid_y, indexing="xy")
    ax.quiver(
        x_mesh,
        y_mesh,
        wind_u[::step, ::step].T,
        wind_v[::step, ::step].T,
        color="black",
        scale=args.arrow_scale,
        width=args.arrow_width,
        alpha=args.arrow_alpha,
        headwidth=args.arrow_headwidth,
        headlength=args.arrow_headlength,
        headaxislength=args.arrow_headaxislength,
        pivot="mid",
    )

    ax.set_title(f"Wind Field (w={wind_seed})", fontsize=args.title_fontsize)
    ax.set_xlabel("Grid X", fontsize=args.axis_fontsize)
    ax.set_ylabel("Grid Y", fontsize=args.axis_fontsize)
    ax.tick_params(axis="both", labelsize=args.tick_fontsize)

    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Wind Speed", fontsize=args.cbar_label_fontsize, labelpad=14)
    cbar.ax.tick_params(labelsize=args.cbar_tick_fontsize)

    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close()


def main():
    args = parse_args()
    catalog_path = normalize_path(args.wind_catalog_json)
    catalog = load_catalog(catalog_path)

    wind_seed = resolve_wind_seed(catalog, args.wind_seed, args.wind_class)
    output_path = args.output_path.format(wind_seed=wind_seed)
    output_path = normalize_path(output_path)

    wind_u, wind_v, start_x, start_y = load_wind_field(catalog, wind_seed)

    save_wind_field_map(
        wind_u=wind_u,
        wind_v=wind_v,
        output_path=output_path,
        wind_seed=wind_seed,
        args=args,
    )

    print(f"Catalog JSON: {catalog_path}")
    print(f"Wind seed: {wind_seed}")
    print(f"Wind subregion start: ({start_x}, {start_y})")
    print(f"Output image: {output_path}")


if __name__ == "__main__":
    main()
