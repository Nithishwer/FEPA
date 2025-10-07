#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging, os, re, math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from fepa.utils.file_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------- config helpers -----------------------
def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative templates in the test config (no file writes)."""
    keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]
    out = dict(config)
    for k in keys:
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out


# ----------------------- plotting helpers -----------------------
def plot_sidechain_distribution(
    df: pd.DataFrame,
    ensembles: Iterable[str],
    output_file: str = "ensemble_histograms.png",
    ncols: int = 4,
    colorby: str = "ensemble",
):
    """
    Plot histograms for each CHI* column, colored by `colorby`. If exactly two
    ensembles are provided, sort columns by decreasing Jensen–Shannon divergence.
    Safely no-ops if there are no CHI columns to plot.
    """
    if not isinstance(ensembles, (list, tuple, set)):
        raise ValueError("`ensembles` must be a list/tuple/set of ensemble names")

    ensembles = list(ensembles)
    if len(ensembles) < 1:
        logging.warning("No ensembles provided to plot_sidechain_distribution; skipping %s", output_file)
        return

    if colorby not in df.columns:
        logging.warning("Column '%s' not found; cannot color histograms. Skipping %s.", colorby, output_file)
        return

    filtered_df = df[df[colorby].isin(ensembles)].copy()
    torsion_columns = [col for col in filtered_df.columns if col.startswith("CHI")]

    if not torsion_columns:
        logging.warning("No CHI* columns to plot for ensembles=%s; skipping %s", ensembles, output_file)
        return

    # JS sorting when comparing exactly two ensembles
    if len(ensembles) == 2:
        js_scores = {}
        e1, e2 = ensembles
        for col in torsion_columns:
            d1 = filtered_df[filtered_df[colorby] == e1][col].dropna().to_numpy()
            d2 = filtered_df[filtered_df[colorby] == e2][col].dropna().to_numpy()
            if d1.size == 0 or d2.size == 0:
                js_scores[col] = 0.0
                continue
            bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=30)
            p1, _ = np.histogram(d1, bins=bins, density=True)
            p2, _ = np.histogram(d2, bins=bins, density=True)
            p1 = p1 + 1e-12
            p2 = p2 + 1e-12
            p1 /= p1.sum() if p1.sum() else 1.0
            p2 /= p2.sum() if p2.sum() else 1.0
            js_scores[col] = jensenshannon(p1, p2, base=2.0)
        torsion_columns = sorted(torsion_columns, key=lambda k: js_scores[k], reverse=True)

    n_plots = len(torsion_columns)
    ncols = max(1, int(ncols))
    nrows = max(1, math.ceil(n_plots / ncols))

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.ravel(axes)

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
        if len(ensembles) == 2:
            title = f"{col}" if 'js_scores' not in locals() else f"{col}; JS={js_scores.get(col, 0.0):.3f}"
        else:
            title = col
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Remove unused subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300)
    finally:
        plt.close()


def plot_sidechain_evolution(
    df: pd.DataFrame,
    ensemble_list: Iterable[str],
    figsize=(20, 15),
    max_cols: int = 4,
    save_path: Optional[str] = None,
):
    """
    Plot time evolution for each CHI* column for the requested ensembles.
    Safely no-ops if no CHI columns.
    """
    if "timestep" not in df.columns or "ensemble" not in df.columns:
        logging.warning("Missing 'timestep' and/or 'ensemble' in dataframe; skipping evolution plot %s", save_path)
        return

    chi_cols = [col for col in df.columns if col.startswith("CHI")]
    if not chi_cols:
        logging.warning("No CHI* columns to plot for evolution; skipping %s", save_path or "<show>")
        return

    ncols = max(1, min(len(chi_cols), max_cols))
    nrows = max(1, math.ceil(len(chi_cols) / ncols))

    ensemble_list = list(ensemble_list)
    if not ensemble_list:
        logging.warning("Empty ensemble_list; skipping evolution plot %s", save_path or "<show>")
        return

    df_filtered = df[df["ensemble"].isin(ensemble_list)]
    palette = sns.color_palette("tab10", len(ensemble_list))
    color_dict = dict(zip(ensemble_list, palette))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.ravel(axes)

    for i, chi in enumerate(chi_cols):
        ax = axes[i]
        for ens in ensemble_list:
            sub_df = df_filtered[df_filtered["ensemble"] == ens]
            if "timestep" in sub_df and chi in sub_df:
                ax.plot(sub_df["timestep"], sub_df[chi], label=ens, color=color_dict[ens])
        ax.set_title(chi, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True)

    # Turn off unused subplots
    last_idx = min(len(chi_cols), len(axes)) - 1
    for j in range(last_idx + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(ensemble_list), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    try:
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
    finally:
        plt.close(fig)


# ----------------------- main analysis -----------------------
def main():
    """Run the analysis using precomputed features CSVs (golden outputs) with robust fallbacks."""
    repo_root = Path(__file__).resolve().parents[3]

    # Load config (kept relative in the repo)
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    config = load_config(str(cfg_path))
    config = _abspath_templates(config, repo_root)

    # Outputs live alongside the golden CSVs
    analysis_output_dir = (repo_root / "tests" / "test_data" / "5_expected").resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Fail-safe: ensure compounds list is iterable
    compounds = list(map(str, config.get("compounds", [])))
    if not compounds:
        logging.error("No compounds specified in %s", cfg_path)
        return

    for cmp in compounds:
        logging.info("Analyzing compound %s ...", cmp)

        # Per-compound output dir (and golden CSV location)
        cmp_output_dir = analysis_output_dir / cmp
        cmp_output_dir.mkdir(parents=True, exist_ok=True)

        # Read golden features
        features_csv = cmp_output_dir / "SideChainTorsions_features.csv"
        if not features_csv.exists():
            logging.error("Features CSV not found for cmp=%s: %s", cmp, features_csv)
            continue

        features_df = pd.read_csv(features_csv)

        # Fail-safe: validate required columns
        for need in ("timestep", "ensemble"):
            if need not in features_df.columns:
                logging.error("Required column '%s' missing in %s; skipping cmp=%s", need, features_csv, cmp)
                continue

        # Quick peek for debugging
        print(features_df.head())

        # Offset mapping:
        # CSV columns carry RAW residue ids in the header like "CHI1 ALA 10".
        # We filter using RAW ids and rename with WITH-OFFSET ids for output.
        offset = 567

        # WITH-OFFSET ids of interest (report space). Add more as needed.
        residues_with_offset = [
            810, 655, 625, 628, 654, 658, 659, 809, 806, 785, 788, 802, 805, 744, 740, 781,
            # If your CSV currently contains residues like 10, 20 (RAW),
            # optionally include their with-offset ids here (10+567=577, 20+567=587):
            # 577, 587,
        ]
        residues_raw_target = {r - offset for r in residues_with_offset if (r - offset) > 0}

        def get_resid(colname: str) -> Optional[int]:
            parts = colname.split()
            if len(parts) >= 4 and parts[3].isdigit():
                return int(parts[3])  # RAW residue id in the CSV header
            return None

        def add_offset(colname: str, offset_val: int = 0) -> str:
            parts = colname.split()
            if len(parts) == 4 and parts[3].isdigit():
                resid = int(parts[3])
                return f"{parts[0]} {parts[1]} {parts[2]} {resid + offset_val}"
            return colname

        # Build filtered view by RAW ids; always include time/ensemble
        chi_cols_filtered = [
            c for c in features_df.columns
            if c.startswith("CHI") and (get_resid(c) in residues_raw_target)
        ]
        base_cols = [c for c in ("timestep", "ensemble") if c in features_df.columns]
        filtered_cols = chi_cols_filtered + base_cols

        filtered_features_df = features_df[filtered_cols].copy()

        # Fail-safe fallback: if ROI yields no CHI columns, use all CHIs so we still produce plots
        if not any(c.startswith("CHI") for c in filtered_features_df.columns):
            logging.warning(
                "ROI with-offset %s (raw %s) not found in this CSV; "
                "falling back to all CHI columns for cmp=%s",
                sorted(residues_with_offset),
                sorted(residues_raw_target),
                cmp,
            )
            chi_cols_all = [c for c in features_df.columns if c.startswith("CHI")]
            filtered_features_df = features_df[chi_cols_all + base_cols].copy()

        # Rename CHI headers to WITH-OFFSET ids for presentation/output
        filtered_features_df.columns = [add_offset(c, offset_val=offset) for c in filtered_features_df.columns]

        # Save filtered CSV deterministically
        out_csv = cmp_output_dir / f"{cmp}_filtered_SideChainTorsions_features.csv"
        filtered_features_df.to_csv(out_csv, index=False)

        # --- Plots ---
        # Distributions
        plot_sidechain_distribution(
            df=filtered_features_df,
            ensembles=["apo_1"],
            output_file=str(cmp_output_dir / f"{cmp}_sidechain_histograms_apo.png"),
            ncols=4,
        )
        plot_sidechain_distribution(
            df=filtered_features_df,
            ensembles=[f"{cmp}_van_1_vdw.20"],
            output_file=str(cmp_output_dir / f"{cmp}_sidechain_histograms_vdw20.png"),
            ncols=4,
        )

        # Time evolution
        plot_sidechain_evolution(
            df=filtered_features_df,
            ensemble_list=["apo_1"],
            figsize=(20, 15),
            max_cols=4,
            save_path=str(cmp_output_dir / f"{cmp}_sidechain_evolution_apo.png"),
        )
        plot_sidechain_evolution(
            df=filtered_features_df,
            ensemble_list=[f"{cmp}_van_1_vdw.20"],
            figsize=(20, 15),
            max_cols=4,
            save_path=str(cmp_output_dir / f"{cmp}_sidechain_evolution_vdw20.png"),
        )
        plot_sidechain_evolution(
            df=filtered_features_df,
            ensemble_list=[f"{cmp}_van_1"],
            figsize=(20, 15),
            max_cols=4,
            save_path=str(cmp_output_dir / f"{cmp}_sidechain_evolution_vanilla.png"),
        )

        # Sim-type overlays
        def get_sim_type(ensemble: str) -> str:
            if "apo" in ensemble:
                return "apo"
            if "coul" in ensemble or "vdw" in ensemble or "rest" in ensemble:
                return "abfe" if re.search(r"van_(\d+)", ensemble) else "abfe"  # lenient
            return "holo"

        enriched = filtered_features_df.copy()
        if "ensemble" in enriched.columns:
            enriched["sim_type"] = enriched["ensemble"].apply(get_sim_type)

            plot_sidechain_distribution(
                df=enriched,
                ensembles=["abfe", "apo"],
                colorby="sim_type",
                output_file=str(cmp_output_dir / f"{cmp}_sidechain_histograms_simtype_abfe_apo.png"),
                ncols=4,
            )
            plot_sidechain_distribution(
                df=enriched,
                ensembles=["holo", "apo"],
                colorby="sim_type",
                output_file=str(cmp_output_dir / f"{cmp}_sidechain_histograms_simtype_holo_apo.png"),
                ncols=4,
            )
        else:
            logging.warning("Column 'ensemble' missing; skipping sim_type overlays for cmp=%s", cmp)


if __name__ == "__main__":
    main()
