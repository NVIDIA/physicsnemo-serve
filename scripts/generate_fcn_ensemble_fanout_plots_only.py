# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


DEFAULT_OUTPUT = Path(
    "outputs/benchmark_report_physicsnemo_serve_fcn_ensemble_fanout_plots_only_1200dpi.png"
)


def generate_report(output: Path, dpi: int) -> None:
    batch = np.array([16, 32, 64])
    labels = [str(value) for value in batch]
    python_wall_seconds = np.array([6687.0, 6959.0, 6688.0])
    physicsnemo_serve_e2e = np.array([269.7, 226.3, 239.6])
    speedup = python_wall_seconds / physicsnemo_serve_e2e
    prepare = np.array([165.7, 142.3, 161.6])
    fanout_execute_collect = np.array([84.0, 73.0, 72.0])
    postprocess_results = np.array([20.0, 11.0, 6.0])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.1,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 10.5,
        }
    )

    fig = plt.figure(figsize=(14.2, 8.2), dpi=dpi, facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.25, 4.0],
        width_ratios=[1.0, 1.0],
        hspace=0.20,
        wspace=0.18,
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.text(
        0.5,
        0.93,
        "FourCastNet (FCN) Ensemble multi-GPU Benchmark",
        ha="center",
        va="top",
        fontsize=30,
        fontweight="bold",
        color="black",
        transform=ax_header.transAxes,
    )
    ax_header.text(
        0.5,
        0.58,
        r"$\bf{Python\ Baseline\ vs\ PhysicsNeMo\ Serve\ Multi\text{-}GPU}$ Wall-Clock Speedup on NVIDIA H100",
        ha="center",
        va="top",
        fontsize=18,
        color="#555555",
        transform=ax_header.transAxes,
    )
    ax_header.text(
        0.5,
        0.37,
        "Python baseline: 1 GPU   |   "
        "PhysicsNeMo Serve multi-GPU: 8 GPUs in parallel with optimized I/O backend",
        ha="center",
        va="top",
        fontsize=12.8,
        color="#666666",
        transform=ax_header.transAxes,
    )
    ax_header.text(
        0.5,
        0.22,
        "custom workflow for 512 ensembles, with 10 steps each   |   "
        "perturbation=gaussian",
        ha="center",
        va="top",
        fontsize=12.8,
        color="#666666",
        transform=ax_header.transAxes,
    )

    ax_speedup = fig.add_subplot(gs[1, 0])
    speedup_colors = ["#a4d65e", "#76b900", "#00a3a3"]
    speedup_edges = ["#76b900", "#4d7f00", "#007a7a"]
    bars = ax_speedup.bar(
        labels,
        speedup,
        width=0.52,
        color=speedup_colors,
        edgecolor=speedup_edges,
        linewidth=1.4,
    )
    ax_speedup.set_title("End-to-End Runtime Speedup", pad=12)
    ax_speedup.set_xlabel("Batch size")
    ax_speedup.set_ylabel("Speedup over Python baseline (x)")
    ax_speedup.set_ylim(0, 35)
    ax_speedup.set_yticks(np.arange(0, 36, 5))
    ax_speedup.grid(axis="y", color="#999999", alpha=0.22, linewidth=0.8)
    for bar, value in zip(bars, speedup, strict=True):
        ax_speedup.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.75,
            f"{value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="black",
        )

    ax_phase = fig.add_subplot(gs[1, 1])
    x = np.arange(len(labels))
    width = 0.56
    phase_colors = {
        "perturbation": "#a8dadc",
        "8 GPU parallel compute": "#43a047",
        "postprocessing": "#4f86a6",
    }
    phase_edges = {
        "perturbation": "#6cc7cf",
        "8 GPU parallel compute": "#2e7d32",
        "postprocessing": "#31627c",
    }
    ax_phase.bar(
        x,
        prepare,
        width,
        label="perturbation",
        color=phase_colors["perturbation"],
        edgecolor=phase_edges["perturbation"],
        linewidth=1.0,
    )
    ax_phase.bar(
        x,
        fanout_execute_collect,
        width,
        bottom=prepare,
        label="8 GPU parallel compute",
        color=phase_colors["8 GPU parallel compute"],
        edgecolor=phase_edges["8 GPU parallel compute"],
        linewidth=1.0,
    )
    ax_phase.bar(
        x,
        postprocess_results,
        width,
        bottom=prepare + fanout_execute_collect,
        label="postprocessing",
        color=phase_colors["postprocessing"],
        edgecolor=phase_edges["postprocessing"],
        linewidth=1.0,
    )
    ax_phase.set_title("PhysicsNeMo Serve Multi-GPU E2E Phase Breakdown", pad=12)
    ax_phase.set_xlabel("Batch size")
    ax_phase.set_ylabel("Duration (seconds)")
    ax_phase.set_xticks(x, labels)
    ax_phase.set_ylim(0, 310)
    ax_phase.set_yticks(np.arange(0, 301, 50))
    ax_phase.grid(axis="y", color="#999999", alpha=0.22, linewidth=0.8)
    ax_phase.legend(loc="upper right", frameon=True, edgecolor="#333333")
    for idx, total in enumerate(physicsnemo_serve_e2e):
        ax_phase.text(
            idx,
            total + 5,
            f"{total:.1f}s",
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color="black",
        )

    fig.subplots_adjust(left=0.065, right=0.975, top=0.965, bottom=0.095)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the PhysicsNeMo Serve FCN ensemble fanout plots-only report PNG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="Output DPI. Defaults to 1200.",
    )
    args = parser.parse_args()

    generate_report(args.output, args.dpi)
    print(args.output)


if __name__ == "__main__":
    main()
