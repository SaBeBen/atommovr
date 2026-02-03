#!/usr/bin/env python3
"""Run the full AtomMover pipeline (generation -> extraction -> PCFA -> AWG).

This script stitches together the components that are already battle-tested in
``test_imaging.py`` (generation, blob extraction, rotation correction,
assignment), ``test_algorithms.py`` (PCFA rearrangement) and
``test_awg_control.py`` (RF conversion). It simulates the experimental data
path, times each processing step except image generation, and produces both a
CSV summary and a compact visualization of the end-to-end result for different
(grid_size, image_shape) configurations.

Requirements from the user story:
    * Blob parameters always come from ``setup_blob_params``.
    * PCFA is the only rearrangement algorithm.
    * ``estimate_grid_rotation_fit_rect`` is the sole rotation estimator.
    * No intermediate visualization; only aggregate figures of the full
      pipeline outcome across configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
# ensure repo root is on sys.path so `import atommover` works when running scripts
_REPO_ROOT = str(_Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml

from atommover.algorithms.single_species import PCFA
from atommover.utils.AtomArray import AtomArray
from atommover.utils.awg_control import AODSettings, RFConverter
from atommover.utils.core import PhysicalParams, array_shape_for_geometry
from atommover.utils.imaging.extraction import (
    BlobDetection,
    fit_grid_and_assign,
    inverse_rotate_centroids,
    estimate_grid_rotation_fit_rect,
)
from atommover.utils.imaging.generation import compute_scaled_image_shape
from atommover.tests.test_imaging import generate_rot_img, setup_blob_params


@dataclass
class PipelineRecord:
    grid_size: int
    image_height: int
    image_width: int
    seed: int
    angle: float
    target_side: int
    target_sites: int
    read_time: float
    cpu_read_time: float
    extraction_time: float
    cpu_extraction_time: float
    rotation_time: float
    cpu_rotation_time: float
    transform_time: float
    cpu_transform_time: float
    assignment_time: float
    cpu_assignment_time: float
    resort_time: float
    cpu_resort_time: float
    awg_time: float
    cpu_awg_time: float
    total_wall_time: float
    total_cpu_time: float
    moves: int
    batches: int
    ramps: int
    awg_duration: float
    assignment_success: bool
    pcfa_success: bool

    def as_dict(self) -> Dict[str, float | int | bool]:
        return self.__dict__.copy()


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_target_side(
    binary: np.ndarray,
    algorithm: object,
    loading_prob: float,
    safety_margin: float = 0.8,
) -> int:
    available = int(binary.sum())
    if available == 0:
        return 0
    max_side = min(binary.shape[0], binary.shape[1])
    geometry_spec = getattr(algorithm, "preferred_geometry_spec", None)
    for candidate in range(max_side, 0, -1):
        if candidate * candidate > available * safety_margin:
            continue
        try:
            pref_rows, pref_cols = array_shape_for_geometry(geometry_spec, candidate, loading_prob=loading_prob)
        except Exception:
            pref_rows, pref_cols = binary.shape
        if pref_rows > binary.shape[0] or pref_cols > binary.shape[1]:
            continue
        return candidate
    return 0


def _build_atom_array(
    binary: np.ndarray,
    phys_params: PhysicalParams,
    algorithm: object,
) -> tuple[AtomArray, np.ndarray, int]:
    rows, cols = binary.shape
    arr = AtomArray([rows, cols], n_species=1, params=phys_params)
    arr.matrix[:, :, 0] = binary.astype(int)

    target_side = _select_target_side(binary, algorithm, phys_params.loading_prob)
    target = np.zeros_like(binary)
    if target_side > 0:
        target[:target_side, :target_side] = 1
    arr.target[:, :, 0] = target
    return arr, target, target_side


def _derive_aod_settings(base_settings: Dict[str, float], grid_size: int, target_side: int) -> AODSettings:
    base_rows = max(2, int(base_settings.get("grid_rows", grid_size)))
    base_cols = max(2, int(base_settings.get("grid_cols", grid_size)))
    f_min_v = float(base_settings.get("f_min_v", 80e6))
    f_max_v = float(base_settings.get("f_max_v", 120e6))
    f_min_h = float(base_settings.get("f_min_h", 80e6))
    f_max_h = float(base_settings.get("f_max_h", 120e6))
    spacing_v = (f_max_v - f_min_v) / max(base_rows - 1, 1)
    spacing_h = (f_max_h - f_min_h) / max(base_cols - 1, 1)
    alignment = base_settings.get("alignment", "center")

    return AODSettings(
        f_min_v=f_min_v,
        f_max_v=f_min_v + spacing_v * max(grid_size - 1, 0),
        f_min_h=f_min_h,
        f_max_h=f_min_h + spacing_h * max(grid_size - 1, 0),
        grid_rows=grid_size,
        grid_cols=grid_size,
        target_rows=target_side,
        target_cols=target_side,
        alignment=alignment,
    )


def _plot_overview(entries: List[Dict], output_path: Path) -> None:
    if not entries:
        return
    _ensure_dir(output_path.parent)
    n_rows = len(entries)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, entry in enumerate(entries):
        img = plt.imread(entry["image_path"])
        axes[row_idx, 0].imshow(img, cmap="Blues")
        axes[row_idx, 0].set_title(f"Input Image\n{entry['label']}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(entry["assigned"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Reconstructed Occupancy")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(entry["final_state"], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("PCFA Target Region")
        axes[row_idx, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_time_vs_target_sites(df: pd.DataFrame, output_path: Path | None) -> None:
    if output_path is None:
        return
    subset = df[df["target_sites"] > 0]
    if subset.empty:
        return
    grouped = (
        subset.groupby("target_sites")[["total_wall_time", "total_cpu_time"]]
        .mean()
        .reset_index()
        .sort_values("target_sites")
    )
    _ensure_dir(output_path.parent)
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette = sns.color_palette("colorblind", 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(
        data=grouped,
        x="target_sites",
        y="total_wall_time",
        marker="o",
        color=palette[0],
        label="Wall time",
        ax=ax,
    )
    sns.lineplot(
        data=grouped,
        x="target_sites",
        y="total_cpu_time",
        marker="s",
        color=palette[1],
        label="CPU time",
        ax=ax,
    )
    ax.fill_between(
        grouped["target_sites"],
        grouped["total_wall_time"],
        alpha=0.08,
        color=palette[0],
    )
    ax.fill_between(
        grouped["target_sites"],
        grouped["total_cpu_time"],
        alpha=0.08,
        color=palette[1],
    )
    ax.set_xlabel("Target sites (atoms)")
    ax.set_ylabel("Wall Time (s)")
    # ax.set_title("Full pipeline scaling")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    # save as svg
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def run_full_pipeline(
    config_path: Path,
    grid_sizes: Sequence[int],
    image_shapes: Sequence[Tuple[int, int]],
    angles: Sequence[float],
    seeds: Sequence[int],
    runs: int,
    output_csv: Path,
    overview_path: Path,
    time_plot_path: Path | None,
) -> None:
    config = _load_config(config_path)
    phys_cfg = config.get("physical_system", {})
    phys_params = PhysicalParams(
        AOD_speed=phys_cfg.get("aod_speed_m_per_us", 0.1),
        spacing=phys_cfg.get("spacing_m", 5e-6),
        loading_prob=phys_cfg.get("loading_prob", 0.6),
        target_occup_prob=phys_cfg.get("target_occup_prob", 0.5),
    )
    awg_base = config.get("awg_control", {}).get("settings", {})

    pcfa = PCFA()

    scratch_dir = config.get("paths", {}).get("scratch_dir", "figs/full_pipeline")
    render_dir = _ensure_dir(Path(scratch_dir) / "full_pipeline")

    records: List[PipelineRecord] = []
    overview_entries: List[Dict] = []

    grid_sizes = list(dict.fromkeys(grid_sizes))
    image_shapes = list(dict.fromkeys(tuple(shape) for shape in image_shapes))
    angles = list(angles)
    seeds = list(seeds)
    if runs <= 0:
        runs = len(seeds)

    for grid_size, image_shape in itertools.product(grid_sizes, image_shapes):
        base_shape = (int(image_shape[0]), int(image_shape[1]))
        shape_tuple = compute_scaled_image_shape(base_shape, grid_size)
        for run_idx in range(min(runs, len(seeds))):
            seed = int(seeds[run_idx])
            angle = float(angles[run_idx % len(angles)])
            np.random.seed(seed)
            suffix = f"pipe_g{grid_size}_h{shape_tuple[0]}w{shape_tuple[1]}_seed{seed}_a{int(angle)}"
            _, true_binary = generate_rot_img(
                image_shape=shape_tuple,
                grid_size=grid_size,
                true_angle=angle,
                suffix=suffix,
                directory=str(render_dir),
            )
            rot_img_path = render_dir / f"{suffix}_rot_image.png"

            # Step timings (generation intentionally excluded)
            total_wall = 0.0
            total_cpu = 0.0

            read_start_wall = time.perf_counter()
            read_start_cpu = time.process_time()
            img = plt.imread(rot_img_path)
            read_time = time.perf_counter() - read_start_wall
            cpu_read_time = time.process_time() - read_start_cpu
            total_wall += read_time
            total_cpu += cpu_read_time

            blob_params = setup_blob_params(
                None,
                image_shape=tuple(int(dim) for dim in img.shape[:2]),
                grid_size=grid_size,
            )
            extractor = BlobDetection(
                shape=(grid_size, grid_size),
                scale=(1, 1),
                logger=None,
                blob_params=blob_params,
            )

            extract_start_wall = time.perf_counter()
            extract_start_cpu = time.process_time()
            centroids, _ = extractor.extract(str(rot_img_path))
            extraction_time = time.perf_counter() - extract_start_wall
            cpu_extraction_time = time.process_time() - extract_start_cpu
            total_wall += extraction_time
            total_cpu += cpu_extraction_time

            rot_start_wall = time.perf_counter()
            rot_start_cpu = time.process_time()
            est_angle = estimate_grid_rotation_fit_rect(centroids, plot=False)
            rotation_time = time.perf_counter() - rot_start_wall
            cpu_rotation_time = time.process_time() - rot_start_cpu
            total_wall += rotation_time
            total_cpu += cpu_rotation_time

            transform_start_wall = time.perf_counter()
            transform_start_cpu = time.process_time()
            centroids_corrected = inverse_rotate_centroids(
                np.asarray(centroids), image_shape=img.shape, angle_deg=est_angle
            )
            transform_time = time.perf_counter() - transform_start_wall
            cpu_transform_time = time.process_time() - transform_start_cpu
            total_wall += transform_time
            total_cpu += cpu_transform_time

            assign_start_wall = time.perf_counter()
            assign_start_cpu = time.process_time()
            assigned = fit_grid_and_assign(
                np.asarray(centroids_corrected),
                (grid_size, grid_size),
                image_shape=img.shape[:2],
            )
            assignment_time = time.perf_counter() - assign_start_wall
            cpu_assignment_time = time.process_time() - assign_start_cpu
            total_wall += assignment_time
            total_cpu += cpu_assignment_time
            assignment_success = bool(np.array_equal(assigned, true_binary))

            arr, target, target_side = _build_atom_array(assigned, phys_params, pcfa)
            resort_start_wall = time.perf_counter()
            resort_start_cpu = time.process_time()
            final_state, move_batches, pcfa_success = pcfa.get_moves(arr, do_ejection=False)
            resort_time = time.perf_counter() - resort_start_wall
            cpu_resort_time = time.process_time() - resort_start_cpu
            total_wall += resort_time
            total_cpu += cpu_resort_time
            move_batches = [list(batch) for batch in move_batches]

            aod_settings = _derive_aod_settings(awg_base, grid_size, target_side)
            converter = RFConverter(aod_settings, phys_params)
            awg_start_wall = time.perf_counter()
            awg_start_cpu = time.process_time()
            awg_sequence = converter.convert_sequence(move_batches)
            awg_time = time.perf_counter() - awg_start_wall
            cpu_awg_time = time.process_time() - awg_start_cpu
            total_wall += awg_time
            total_cpu += cpu_awg_time

            moves = sum(len(batch) for batch in move_batches)
            ramps = sum(len(batch.ramps) for batch in awg_sequence)
            awg_duration = sum(batch.total_duration for batch in awg_sequence)
            target_sites = int(target_side * target_side)

            records.append(
                PipelineRecord(
                    grid_size=grid_size,
                    image_height=shape_tuple[0],
                    image_width=shape_tuple[1],
                    seed=seed,
                    angle=angle,
                    target_side=target_side,
                    target_sites=target_sites,
                    read_time=read_time,
                    cpu_read_time=cpu_read_time,
                    extraction_time=extraction_time,
                    cpu_extraction_time=cpu_extraction_time,
                    rotation_time=rotation_time,
                    cpu_rotation_time=cpu_rotation_time,
                    transform_time=transform_time,
                    cpu_transform_time=cpu_transform_time,
                    assignment_time=assignment_time,
                    cpu_assignment_time=cpu_assignment_time,
                    resort_time=resort_time,
                    cpu_resort_time=cpu_resort_time,
                    awg_time=awg_time,
                    cpu_awg_time=cpu_awg_time,
                    total_wall_time=total_wall,
                    total_cpu_time=total_cpu,
                    moves=moves,
                    batches=len(move_batches),
                    ramps=ramps,
                    awg_duration=awg_duration,
                    assignment_success=assignment_success,
                    pcfa_success=bool(pcfa_success),
                )
            )

            if run_idx == 0:
                overview_entries.append(
                    {
                        "label": f"g={grid_size}, img={shape_tuple[0]}x{shape_tuple[1]}",
                        "image_path": rot_img_path,
                        "assigned": assigned,
                        "final_state": np.asarray(final_state),
                    }
                )

    if not records:
        print("No runs executed; nothing to report.")
        return

    df = pd.DataFrame([rec.as_dict() for rec in records])
    _ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    print(f"Wrote timing results to {output_csv}")

    summary = df.groupby(["grid_size", "image_height", "image_width"]).agg({
        "target_side": ["mean"],
        "target_sites": ["mean"],
        "read_time": ["mean", "std"],
        "cpu_read_time": ["mean", "std"],
        "extraction_time": ["mean", "std"],
        "cpu_extraction_time": ["mean", "std"],
        "rotation_time": ["mean", "std"],
        "cpu_rotation_time": ["mean", "std"],
        "transform_time": ["mean", "std"],
        "cpu_transform_time": ["mean", "std"],
        "assignment_time": ["mean", "std"],
        "cpu_assignment_time": ["mean", "std"],
        "resort_time": ["mean", "std"],
        "cpu_resort_time": ["mean", "std"],
        "awg_time": ["mean", "std"],
        "cpu_awg_time": ["mean", "std"],
        "total_wall_time": ["mean", "std"],
        "total_cpu_time": ["mean", "std"],
        "assignment_success": "mean",
        "pcfa_success": "mean",
    })
    summary_path = output_csv.with_name(output_csv.stem + "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Wrote grouped summary to {summary_path}")

    target_summary = (
        df[df["target_sites"] > 0]
        .groupby("target_sites")
        .agg(
            wall_time_mean=("total_wall_time", "mean"),
            wall_time_std=("total_wall_time", "std"),
            cpu_time_mean=("total_cpu_time", "mean"),
            cpu_time_std=("total_cpu_time", "std"),
        )
    )
    target_summary_path = output_csv.with_name(output_csv.stem + "_target_sites.csv")
    target_summary.to_csv(target_summary_path)
    print(f"Wrote target-site summary to {target_summary_path}")

    _plot_overview(overview_entries, overview_path)
    print(f"Saved overview figure to {overview_path}")
    _plot_time_vs_target_sites(df, time_plot_path)
    if time_plot_path is not None:
        print(f"Saved timing plot to {time_plot_path}")


def _parse_image_shapes(values: Iterable[str]) -> List[Tuple[int, int]]:
    shapes: List[Tuple[int, int]] = []
    for token in values:
        if "x" not in token.lower():
            raise ValueError(f"Image shape '{token}' must look like HxW, e.g. 640x1280")
        h_str, w_str = token.lower().split("x", maxsplit=1)
        shapes.append((int(h_str), int(w_str)))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the full AtomMover pipeline")
    parser.add_argument("--config", type=Path, default=Path("pipeline_config.yaml"), help="YAML pipeline config")
    parser.add_argument("--grid-sizes", type=int, nargs="*", help="Grid sizes to simulate")
    parser.add_argument(
        "--image-shapes",
        nargs="*",
        help="Image shapes as HxW tokens (e.g. 640x1280). Provide multiple to compare.",
    )
    parser.add_argument("--angles", type=float, nargs="*", help="Rotation angles to sample (deg)")
    parser.add_argument("--seeds", type=int, nargs="*", help="Random seeds to reuse for reproducibility")
    parser.add_argument("--runs", type=int, default=0, help="How many runs per configuration (defaults to len(seeds))")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/benchmark_pipeline/full_pipeline_times.csv"),
        help="Where to store timing results",
    )
    parser.add_argument(
        "--overview",
        type=Path,
        default=Path("figs/benchmark_pipeline/full_pipeline/full_pipeline_overview.png"),
        help="Path for the aggregate visualization",
    )
    parser.add_argument(
        "--time-plot",
        type=Path,
        default=None,
        help="Path for the CPU vs wall time plot (overrides config)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    pipeline_cfg = cfg.get("benchmarking", {}).get("pipeline_benchmark", {})
    default_grid_range = cfg.get("imaging", {}).get("sample_generation", {}).get("grid_size_range", [9, 21])
    default_grids = [int(default_grid_range[0]), int(default_grid_range[-1])]
    config_grid_sizes = pipeline_cfg.get("grid_sizes")
    if config_grid_sizes:
        config_grid_sizes = [int(val) for val in config_grid_sizes]
    grid_sizes = args.grid_sizes or config_grid_sizes or default_grids

    default_shape = tuple(cfg.get("imaging", {}).get("sample_generation", {}).get("image_shape", [640, 1280]))
    alt_shape = (max(128, default_shape[0] // 2), default_shape[1])
    if args.image_shapes:
        image_shapes = _parse_image_shapes(args.image_shapes)
    elif pipeline_cfg.get("image_shapes"):
        image_shapes = _parse_image_shapes(pipeline_cfg["image_shapes"])
    else:
        image_shapes = [default_shape, alt_shape]

    default_angles = cfg.get("imaging", {}).get("sample_generation", {}).get("rotation_angles_deg", [-5, 0, 5])
    config_angles = pipeline_cfg.get("angles")
    angles = args.angles or config_angles or default_angles

    default_seeds = cfg.get("imaging", {}).get("sample_generation", {}).get("seeds", [42, 43, 44])
    config_seeds = pipeline_cfg.get("seeds")
    seeds = args.seeds or config_seeds or default_seeds

    config_time_plot = pipeline_cfg.get("time_plot_path")
    time_plot_path = args.time_plot or (Path(config_time_plot) if config_time_plot else None)
    if time_plot_path is None:
        time_plot_path = Path("figs/benchmark_pipeline/full_pipeline/full_pipeline_time_vs_target_sites.png")

    run_full_pipeline(
        config_path=args.config,
        grid_sizes=grid_sizes,
        image_shapes=image_shapes,
        angles=angles,
        seeds=seeds,
        runs=args.runs,
        output_csv=args.output_csv,
        overview_path=args.overview,
        time_plot_path=time_plot_path,
    )


if __name__ == "__main__":
    main()
