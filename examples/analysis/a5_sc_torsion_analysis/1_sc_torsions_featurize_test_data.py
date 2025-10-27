import logging
import os
import pandas as pd
from pathlib import Path
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import (
    load_paths_for_compound,
    load_abfe_paths_for_compound,
    load_paths_for_apo,
)
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from scipy.spatial.distance import jensenshannon
from fepa.utils.dimred_utils import (
    cluster_pca,
    get_ensemble_center,
    make_ensemble_center_df,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative templates in the test config."""
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

def plot_sidechain_distribution(
    df, ensembles, output_file="ensemble_histograms.png", ncols=4, colorby="ensemble"
):
    """
    Plots histograms for each column (excluding 'timestep' and the colorby column) in subplots,
    colored by the ensemble, and saves the entire plot to one image file.

    If exactly two ensembles are provided, columns are ordered by decreasing JS divergence.

    Parameters:
    - df: pandas DataFrame
    - ensembles: list/tuple/set of ensemble names to include (should be length 2 for JS sorting)
    - output_file: file path to save the combined plot
    - ncols: number of columns in the subplot grid
    - colorby: column name to color by (default is 'ensemble')
    """
    if not isinstance(ensembles, (list, tuple, set)):
        raise ValueError("`ensembles` must be a list, tuple, or set of ensemble names")

    ensembles = list(ensembles)
    if len(ensembles) < 1:
        raise ValueError("Must provide at least one ensemble")

    filtered_df = df[df[colorby].isin(ensembles)].copy()
    torsion_columns = [col for col in df.columns if "CHI" in col]

    # Compute JS divergence if exactly two ensembles
    if len(ensembles) == 2:
        js_scores = {}
        e1, e2 = ensembles
        for col in torsion_columns:
            # Drop NA
            d1 = filtered_df[filtered_df[colorby] == e1][col].dropna()
            d2 = filtered_df[filtered_df[colorby] == e2][col].dropna()

            # Shared bin edges
            min_val = min(d1.min(), d2.min())
            max_val = max(d1.max(), d2.max())
            bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=30)

            # Histogram densities
            p1, _ = np.histogram(d1, bins=bins, density=True)
            p2, _ = np.histogram(d2, bins=bins, density=True)

            # Avoid 0s for JS divergence (add small constant)
            p1 += 1e-12
            p2 += 1e-12

            p1 /= p1.sum()
            p2 /= p2.sum()

            js = jensenshannon(p1, p2, base=2.0)
            js_scores[col] = js

        # Sort columns by JS divergence
        torsion_columns = sorted(
            torsion_columns, key=lambda k: js_scores[k], reverse=True
        )

    n_plots = len(torsion_columns)
    nrows = math.ceil(n_plots / ncols)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

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
        if len(ensembles) == 2:
            title += f"{col}; \nJS={js_scores[col]:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Remove unused subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_sidechain_evolution(
    df: pd.DataFrame, ensemble_list: list, figsize=(20, 15), max_cols=4, save_path=None
):
    """
    Plots the time evolution of CHI-related variables for specified ensembles.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        ensemble_list (list): List of ensembles to include in the plot.
        figsize (tuple): Size of the full figure.
        max_cols (int): Maximum number of subplot columns in the grid.

    Returns:
        None
    """
    # Filter only CHI columns
    chi_cols = [col for col in df.columns if "CHI" in col]
    n_chis = len(chi_cols)

    # Determine subplot grid size
    ncols = min(n_chis, max_cols)
    nrows = math.ceil(n_chis / ncols)

    # Filter by ensemble
    df_filtered = df[df["ensemble"].isin(ensemble_list)]

    # Set up color palette
    palette = sns.color_palette("tab10", len(ensemble_list))
    color_dict = dict(zip(ensemble_list, palette))

    # Start plotting
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

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(ensemble_list),
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300) if save_path else plt.show()
    plt.close(fig)


def main():
    """Main function to run the analysis"""

    repo_root = Path(__file__).resolve().parents[3]

    # Load config (kept relative in the repo)
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    config = load_config(str(cfg_path))

    # Make path templates absolute w.r.t. repo root (no change to the JSON file itself)
    config = _abspath_templates(config, repo_root)

    # Where to write expected outputs
    analysis_output_dir = (repo_root / "tests" / "test_data" / "5_expected").resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    for cmp in config["compounds"][:]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=[1],
            leg_window_list = [f"coul.{i:02}" for i in range(2)], # [f"vdw.{i:02d}" for i in range(20, 21)],
            # + [f"coul.{i:02d}" for i in range(0, 11)]
            # + [f"rest.{i:02d}" for i in range(0, 12)],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

        # Used prod_fixed.xtc instead of prod.xtc for apo sims
        # prod_fixed.xtc is the same as prod.xtc but with protein whole and pbc fixed
        # for key, value in path_dict.items():
        #     if "apo" in key:
        #         path_dict[key]["xtc"] = value["xtc"].replace(
        #             "prod.xtc", "prod_fixed.xtc"
        #         )

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()

        ## Featurize
        logging.info("Featurizing binding pocket waters ...")
        sct_featurizer = SideChainTorsionsFeaturizer(ensemble_handler=ensemble_handler)

        sct_featurizer.featurize()

        # Save features
        logging.info("Saving features for compound %s ...", cmp)
        sct_featurizer.save_features(cmp_output_dir, overwrite=True)

        # features_df = pd.read_csv(
        #     os.path.join(cmp_output_dir, f"SideChainTorsions_features.csv")
        # )

        # # Get only the aminoacid residues of interest
        # residues_of_interest = [
        #     853,
        #     843,
        #     278,
        #     277,
        #     270,
        #     253,
        #     250,
        #     249,
        #     246,
        #     181,
        #     177,
        #     168,
        #     66,
        #     63,
        #     13,
        # ]

        # # Function to extract residue number
        # def get_resid(colname):
        #     parts = colname.split()
        #     if len(parts) >= 4 and parts[3].isdigit():
        #         return int(parts[3])
        #     return None

        # # Filter columns whose residue number is in resid_list
        # filtered_cols = [
        #     col for col in features_df.columns if get_resid(col) in residues_of_interest
        # ] + ["timestep", "ensemble"]

        # # Create a new DataFrame with the filtered columns
        # filtered_features_df = features_df[filtered_cols]

        # # Function to get the simulation type
        # def get_sim_type(ensemble):
        #     if "apo" in ensemble:
        #         return "apo"
        #     if "coul" in ensemble or "vdw" in ensemble or "rest" in ensemble:
        #         match = re.search(r"van_(\d+)", ensemble)
        #         if match:
        #             return f"abfe"
        #         else:
        #             raise ValueError(f"van not found in ensemble: {ensemble}")
        #     else:
        #         return "holo"

        # # Save the filtered DataFrame
        # filtered_features_df.to_csv(
        #     os.path.join(
        #         cmp_output_dir, f"{cmp}_filtered_SideChainTorsions_features.csv"
        #     ),
        #     index=False,
        # )

        # # Plot the distributions
        # plot_sidechain_distribution(
        #     df=filtered_features_df,
        #     ensembles=[
        #         "apo_1",
        #         "apo_2",
        #         "apo_3",
        #     ],
        #     output_file=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_histograms_apo.png"
        #     ),
        #     ncols=4,
        # )

        # plot_sidechain_distribution(
        #     df=filtered_features_df,
        #     ensembles=[
        #         f"{cmp}_van_1_vdw.20",
        #         f"{cmp}_van_2_vdw.20",
        #         f"{cmp}_van_3_vdw.20",
        #     ],
        #     output_file=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_histograms_vdw20.png"
        #     ),
        #     ncols=4,
        # )

        # # Plot the time evolution of CHI-related variables
        # plot_sidechain_evolution(
        #     df=filtered_features_df,
        #     ensemble_list=[
        #         "apo_1",
        #         "apo_2",
        #         "apo_3",
        #     ],
        #     figsize=(20, 15),
        #     max_cols=4,
        #     save_path=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_evolution_apo.png"
        #     ),
        # )

        # plot_sidechain_evolution(
        #     df=filtered_features_df,
        #     ensemble_list=[
        #         f"{cmp}_van_1_vdw.20",
        #         f"{cmp}_van_2_vdw.20",
        #         f"{cmp}_van_3_vdw.20",
        #     ],
        #     figsize=(20, 15),
        #     max_cols=4,
        #     save_path=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_evolution_vdw20.png"
        #     ),
        # )

        # plot_sidechain_evolution(
        #     df=filtered_features_df,
        #     ensemble_list=[
        #         f"{cmp}_van_1",
        #         f"{cmp}_van_2",
        #         f"{cmp}_van_3",
        #     ],
        #     figsize=(20, 15),
        #     max_cols=4,
        #     save_path=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_evolution_vanilla.png"
        #     ),
        # )

        # # Annotate sim_type for features_df
        # filtered_features_df["sim_type"] = filtered_features_df["ensemble"].apply(
        #     get_sim_type
        # )

        # plot_sidechain_distribution(
        #     df=filtered_features_df,
        #     ensembles=["abfe", "apo"],
        #     colorby="sim_type",
        #     output_file=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_histograms_simtype_abfe_apo.png"
        #     ),
        #     ncols=4,
        # )

        # plot_sidechain_distribution(
        #     df=filtered_features_df,
        #     ensembles=["holo", "apo"],
        #     colorby="sim_type",
        #     output_file=os.path.join(
        #         cmp_output_dir, f"{cmp}_sidechain_histograms_simtype_holo_apo.png"
        #     ),
        #     ncols=4,
        # )


if __name__ == "__main__":
    main()
