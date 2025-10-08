#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create golden CSVs for side-chain torsions features in tests/test_data/5_data.

Behavior:
- Uses tests/test_config/config.json
- For each compound:
  - If tests/test_data/5_input/<cmp>/SideChainTorsions_features.csv exists, copies it.
  - Else, synthesizes a small deterministic CSV suitable for exact pytest comparison.
- Optionally produces quick-look plots when FEPA_SCT_PLOTS=1.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt
import seaborn as sns

from fepa.utils.file_utils import load_config  # only FEPA import we need

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------- plotting helpers (used only if FEPA_SCT_PLOTS=1) ----------
def plot_sidechain_distribution(
    df: pd.DataFrame,
    ensembles,
    output_file: Path,
    ncols: int = 4,
    colorby: str = "ensemble",
):
    if not isinstance(ensembles, (list, tuple, set)):
        raise ValueError("`ensembles` must be a list, tuple, or set of ensemble names")
    ensembles = list(ensembles)
    if len(ensembles) < 1:
        raise ValueError("Must provide at least one ensemble")

    filtered_df = df[df[colorby].isin(ensembles)].copy()
    torsion_columns = [c for c in df.columns if "CHI" in c]

    # If exactly two ensembles, rank by Jensen-Shannon divergence to make plots informative
    if len(ensembles) == 2:
        from scipy.spatial.distance import jensenshannon
        js_scores = {}
        e1, e2 = ensembles
        for col in torsion_columns:
            d1 = filtered_df[filtered_df[colorby] == e1][col].dropna()
            d2 = filtered_df[filtered_df[colorby] == e2][col].dropna()
            if d1.empty or d2.empty:
                js_scores[col] = 0.0
                continue
            bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=30)
            p1, _ = np.histogram(d1, bins=bins, density=True)
            p2, _ = np.histogram(d2, bins=bins, density=True)
            p1 = (p1 + 1e-12) / (p1.sum() + 1e-12)
            p2 = (p2 + 1e-12) / (p2.sum() + 1e-12)
            js_scores[col] = float(jensenshannon(p1, p2, base=2.0))
        torsion_columns = sorted(torsion_columns, key=lambda k: js_scores.get(k, 0.0), reverse=True)
    else:
        js_scores = {}

    n_plots = len(torsion_columns)
    ncols = max(1, min(ncols, n_plots)) if n_plots else 1
    nrows = math.ceil(max(1, n_plots) / ncols)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, col in enumerate(torsion_columns):
        ax = axes[idx]
        sns.histplot(
            data=filtered_df,
            x=col,
            hue=colorby,
            kde=True,
            stat="density",
            common_norm=False,
            bins=30,
            ax=ax,
        )
        title = col
        if len(ensembles) == 2 and col in js_scores:
            title += f"\nJS={js_scores[col]:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Remove unused subplots
    for k in range(n_plots, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_sidechain_evolution(
    df: pd.DataFrame,
    ensemble_list: list[str],
    save_path: Path,
    figsize=(20, 15),
    max_cols: int = 4,
):
    chi_cols = [c for c in df.columns if "CHI" in c]
    n_chis = len(chi_cols)
    if n_chis == 0:
        # Create an empty figure to keep CI logic simple
        plt.figure(figsize=(6, 4))
        plt.title("No CHI columns")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    ncols = min(n_chis, max_cols)
    nrows = math.ceil(n_chis / ncols)
    df_filtered = df[df["ensemble"].isin(ensemble_list)]
    palette = sns.color_palette("tab10", len(ensemble_list))
    color_dict = dict(zip(ensemble_list, palette))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i, chi in enumerate(chi_cols):
        ax = axes[i]
        for ens in ensemble_list:
            sub = df_filtered[df_filtered["ensemble"] == ens]
            ax.plot(sub["timestep"], sub[chi], label=ens, color=color_dict[ens])
        ax.set_title(chi, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[min(i, len(axes) - 1)].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ensemble_list), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------- core helpers ----------
def _repo_root() -> Path:
    # This file: FEPA/scripts/... -> parents[3] == FEPA/
    return Path(__file__).resolve().parents[3]


def _load_test_config(repo_root: Path) -> dict:
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    return load_config(str(cfg_path))


def _input_features_path(repo_root: Path, cmp: str | int) -> Path:
    return repo_root / "tests" / "test_data" / "5_input" / str(cmp) / "SideChainTorsions_features.csv"


def _output_dir(repo_root: Path, cmp: str | int) -> Path:
    return repo_root / "tests" / "test_data" / "5_data" / str(cmp)


def _output_features_path(repo_root: Path, cmp: str | int) -> Path:
    return _output_dir(repo_root, cmp) / "SideChainTorsions_features.csv"


def _synthesize_features(cmp: str | int, n: int = 60) -> pd.DataFrame:
    """
    Make a tiny deterministic side-chain torsions dataset suitable for exact CSV comparison.
    - Two ensembles: apo_1 and <cmp>_van_1_vdw.20
    - CHI angles in radians (or degrees—tests don't care; be consistent)
    """
    np.random.seed(0)
    ts = np.arange(n, dtype=int)
    ens = np.array(["apo_1"] * n + [f"{cmp}_van_1_vdw.20"] * n)
    timestep = np.concatenate([ts, ts])

    # Two CHI columns with separable means
    chi1 = np.concatenate([np.random.normal(-1.0, 0.15, n), np.random.normal(1.0, 0.15, n)])
    chi2 = np.concatenate([np.random.normal(0.0, 0.20, n), np.random.normal(0.2, 0.20, n)])

    df = pd.DataFrame({
        "timestep": timestep,
        "ensemble": ens,
        "CHI1 ALA 10": chi1,
        "CHI2 TYR 20": chi2,
    })
    return df


def _ensure_sidechain_features(repo_root: Path, cmp: str | int, test_mode: bool) -> Path:
    """
    Ensure tests/test_data/5_data/<cmp>/SideChainTorsions_features.csv exists.
    Priority:
      1) If tests/test_data/5_input/<cmp>/SideChainTorsions_features.csv exists -> copy
      2) If FEPA_TEST_MODE=1 -> synthesize deterministic CSV
      3) Else: error (to avoid pulling trajectories in CI)
    """
    in_f = _input_features_path(repo_root, cmp)
    out_d = _output_dir(repo_root, cmp)
    out_f = _output_features_path(repo_root, cmp)
    out_d.mkdir(parents=True, exist_ok=True)

    if in_f.exists():
        df = pd.read_csv(in_f)
        df.to_csv(out_f, index=False)
        logging.info("Copied features -> %s", out_f)
        return out_f

    if test_mode:
        df = _synthesize_features(cmp)
        df.to_csv(out_f, index=False)
        logging.info("Synthesized features -> %s", out_f)
        return out_f

    raise FileNotFoundError(
        f"Missing input features {in_f}. "
        f"Either provide test input or set FEPA_TEST_MODE=1 to synthesize."
    )


# ---------- main ----------
def main():
    """
    Produce golden CSVs for test verification at tests/test_data/5_data/.
    This script does NOT run MDAnalysis in CI; it either copies provided inputs or synthesizes small tables.
    """
    repo_root = _repo_root()
    config = _load_test_config(repo_root)

    test_mode = os.environ.get("FEPA_TEST_MODE", "1") in ("1", "true", "True")
    make_plots = os.environ.get("FEPA_SCT_PLOTS", "0") in ("1", "true", "True")

    compounds = list(map(str, config.get("compounds", [])))
    if not compounds:
        logging.warning("No compounds in test config. Nothing to do.")
        return

    for cmp in compounds:
        logging.info("Preparing side-chain torsions features for compound=%s", cmp)
        feat_csv = _ensure_sidechain_features(repo_root, cmp, test_mode=test_mode)

        if make_plots:
            df = pd.read_csv(feat_csv)
            out_dir = _output_dir(repo_root, cmp)

            # Apo vs vdw.20 if available
            apo_like = [x for x in df["ensemble"].unique() if str(x).startswith("apo")]
            vdw_like = [x for x in df["ensemble"].unique() if "vdw.20" in str(x)]

            if apo_like:
                # Distribution
                plot_sidechain_distribution(
                    df=df,
                    ensembles=apo_like[:2] if len(apo_like) >= 2 else apo_like,
                    output_file=out_dir / "sidechain_histograms_apo.png",
                    ncols=4,
                    colorby="ensemble",
                )
                # Evolution
                plot_sidechain_evolution(
                    df=df,
                    ensemble_list=apo_like,
                    save_path=out_dir / "sidechain_evolution_apo.png",
                )

            if vdw_like:
                plot_sidechain_distribution(
                    df=df,
                    ensembles=vdw_like[:2] if len(vdw_like) >= 2 else vdw_like,
                    output_file=out_dir / "sidechain_histograms_vdw20.png",
                    ncols=4,
                    colorby="ensemble",
                )
                plot_sidechain_evolution(
                    df=df,
                    ensemble_list=vdw_like,
                    save_path=out_dir / "sidechain_evolution_vdw20.png",
                )

    logging.info("Done. Golden CSV(s) written under tests/test_data/5_data/.")


if __name__ == "__main__":
    main()
