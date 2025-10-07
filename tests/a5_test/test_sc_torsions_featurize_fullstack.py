import logging
import os
from pathlib import Path
import re
import math

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import (
    load_abfe_paths_for_compound,
    load_paths_for_compound,
    load_paths_for_apo,
)
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.utils.dimred_utils import (
    cluster_pca,
    get_ensemble_center,
    make_ensemble_center_df,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# plotting helpers (unchanged)
# ---------------------------

def plot_sidechain_distribution(
    df, ensembles, output_file="ensemble_histograms.png", ncols=4, colorby="ensemble"
):
    if not isinstance(ensembles, (list, tuple, set)):
        raise ValueError("`ensembles` must be a list, tuple, or set of ensemble names")

    ensembles = list(ensembles)
    if len(ensembles) < 1:
        raise ValueError("Must provide at least one ensemble")

    filtered_df = df[df[colorby].isin(ensembles)].copy()
    torsion_columns = [col for col in df.columns if "CHI" in col]

    if len(ensembles) == 2:
        js_scores = {}
        e1, e2 = ensembles
        for col in torsion_columns:
            d1 = filtered_df[filtered_df[colorby] == e1][col].dropna()
            d2 = filtered_df[filtered_df[colorby] == e2][col].dropna()
            bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=30)
            p1, _ = np.histogram(d1, bins=bins, density=True)
            p2, _ = np.histogram(d2, bins=bins, density=True)
            p1 = (p1 + 1e-12) / max(p1.sum(), 1e-12)
            p2 = (p2 + 1e-12) / max(p2.sum(), 1e-12)
            js_scores[col] = jensenshannon(p1, p2, base=2.0)
        torsion_columns = sorted(torsion_columns, key=lambda k: js_scores[k], reverse=True)

    n_plots = len(torsion_columns)
    nrows = math.ceil(n_plots / ncols)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(torsion_columns):
        ax = axes[idx]
        sns.histplot(
            data=filtered_df, x=col, hue=colorby, kde=True,
            stat="density", common_norm=False, bins=30, ax=ax,
        )
        title = col
        if len(ensembles) == 2:
            title += f"{col}; \nJS={js_scores[col]:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_sidechain_evolution(
    df: pd.DataFrame, ensemble_list: list, figsize=(20, 15), max_cols=4, save_path=None
):
    chi_cols = [col for col in df.columns if "CHI" in col]
    n_chis = len(chi_cols)
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
            sub_df = df_filtered[df_filtered["ensemble"] == ens]
            ax.plot(sub_df["timestep"], sub_df[chi], label=ens, color=color_dict[ens])
        ax.set_title(chi, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ensemble_list), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close(fig)

# ---------------------------
# minimal-data friendly utils
# ---------------------------

def _ensure_config_keys(config: dict) -> dict:
    """Add backward-compat keys expected by path_utils if missing."""
    if "vanilla_path_template_old" not in config:
        config["vanilla_path_template_old"] = config["vanilla_path_template"]
    return config

def _exists_any(path: Path, names: list[str]) -> bool:
    return any((path / n).exists() for n in names)

def _detect_available_vanilla_reps(config: dict, cmp_name: str) -> list[int]:
    """Return [1] if only vanilla_rep_1 exists; otherwise return the subset of [1,2,3] that exist."""
    reps = []
    for rep in (1, 2, 3):
        vdir = Path(config["vanilla_path_template"].format(CMP_NAME=cmp_name, REP_NO=rep))
        if _exists_any(vdir, ["npt.gro", "nvt.tpr", "prod.tpr"]):
            reps.append(rep)
    if not reps:
        return [1]  # still try with 1; loader will error if truly absent
    return reps

def _detect_available_leg_windows(config: dict, cmp_name: str) -> list[str]:
    """
    Probe for leg windows we actually have. Accept both naming schemes:
      - prod.tpr/prod.xtc
      - topol.tpr/traj_comp.xtc
    """
    candidates = ["vdw.20", "coul.00", "coul.01"]
    have = []
    for leg in candidates:
        p = Path(config["abfe_window_path_template"].format(CMP_NAME=cmp_name, LEG_WINDOW=leg))
        tpr_ok = _exists_any(p, ["prod.tpr", "topol.tpr"])
        xtc_ok = _exists_any(p, ["prod.xtc", "traj_comp.xtc"])
        if tpr_ok and xtc_ok:
            have.append(leg)
    return have

def _apo_is_available(config: dict, cmp_name: str) -> bool:
    apo_dir = Path(config["apo_path_template"])
    return _exists_any(apo_dir, ["prod.tpr"]) and _exists_any(apo_dir, ["prod.xtc", "prod_fixed.xtc"]) and (apo_dir / "npt.gro").exists()

def _swap_apo_fixed_xtc_if_present(path_dict: dict):
    """Replace apo xtc to prod_fixed.xtc when present (script had this commented-out option)."""
    for key, val in path_dict.items():
        if "apo" in key:
            xtc = Path(val["xtc"])
            fixed = xtc.parent / "prod_fixed.xtc"
            if fixed.exists():
                val["xtc"] = str(fixed)

# ---------------------------
# main (minimal-data aware)
# ---------------------------

def main():
    """Main function to run the analysis on whatever minimal data is present."""
    # Load configuration from the example’s relative path
    config_path = os.path.join("../../tests/test_config/config.json")
    config = load_config(config_path)
    config = _ensure_config_keys(config)

    analysis_output_dir = "wdir"
    os.makedirs(analysis_output_dir, exist_ok=True)

    # IMPORTANT: do not slice [1:] blindly — process what exists
    compounds = list(map(str, config.get("compounds", [])))
    if not compounds:
        logging.warning("No compounds in config; nothing to do.")
        return

    # If user followed older example with dummy first entry, skip it
    if compounds[0] == "__DUMMY__" and len(compounds) >= 2:
        compounds = compounds[1:]

    for cmp in compounds:
        logging.info("Analyzing compound %s ...", cmp)

        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Find what we truly have on disk
        van_list = _detect_available_vanilla_reps(config, cmp)
        leg_window_list = _detect_available_leg_windows(config, cmp)
        apo_flag = _apo_is_available(config, cmp)

        if not leg_window_list and not apo_flag:
            logging.error(
                "No usable leg windows and no APO data found for cmp=%s. "
                "Checked candidates: vdw.20, coul.00, coul.01; and apo dir.",
                cmp,
            )
            continue

        logging.info(
            "Loading paths for compound %s (van_list=%s, legs=%s, apo=%s)...",
            cmp, van_list, leg_window_list, apo_flag
        )

        # Build the binding-pocket CA selection from config
        bp_sel = "name CA and resid " + config["pocket_residues_string"]

        # Ask FEPA to resolve paths exactly as in the original pipeline
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=van_list,
            leg_window_list=leg_window_list or [],  # FEPA handles empty OK if apo=True
            bp_selection_string=bp_sel,
            apo=apo_flag,
        )

        # If apo has a prod_fixed.xtc, switch to it (optional QoL)
        _swap_apo_fixed_xtc_if_present(path_dict)

        # Load trajectories with the real EnsembleHandler
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()

        # Real featurizer
        logging.info("Featurizing side-chain torsions ...")
        sct_featurizer = SideChainTorsionsFeaturizer(ensemble_handler=ensemble_handler)
        sct_featurizer.featurize()

        # Save features
        logging.info("Saving features for compound %s ...", cmp)
        sct_featurizer.save_features(cmp_output_dir, overwrite=True)

        # --- OPTIONAL plots: comment out in CI if unwanted ---
        # features_df = pd.read_csv(os.path.join(cmp_output_dir, "SideChainTorsions_features.csv"))
        # plot_sidechain_distribution(
        #     df=features_df,
        #     ensembles=[ens for ens in features_df["ensemble"].unique() if "apo" in ens][:2],
        #     output_file=os.path.join(cmp_output_dir, f"{cmp}_sidechain_histograms_apo.png"),
        #     ncols=4,
        # )
        # plot_sidechain_evolution(
        #     df=features_df,
        #     ensemble_list=[ens for ens in features_df["ensemble"].unique() if "_vdw.20" in ens][:3],
        #     figsize=(20, 15),
        #     max_cols=4,
        #     save_path=os.path.join(cmp_output_dir, f"{cmp}_sidechain_evolution_vdw20.png"),
        # )

if __name__ == "__main__":
    main()
