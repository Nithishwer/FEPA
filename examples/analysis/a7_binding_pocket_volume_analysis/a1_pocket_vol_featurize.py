"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import BindingPocketVolumeFeaturizer

# from fepa.features.featurizes import BindingPocketVolumeFeaturizer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.utils.file_utils import load_config
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
from fepa.utils.path_utils import load_paths_for_compound, load_abfe_paths_for_compound
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("wdir")

    for cmp in config["compounds"][1:]:
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
            van_list=[1, 2, 3],
            leg_window_list=[
                # "coul.00",
                # "coul.01",
                # "coul.02",
                # "coul.03",
                # "coul.04",
                # "coul.05",
                # "coul.06",
                # "coul.07",
                # "coul.08",
                # "coul.09",
                # "coul.10",
                "vdw.00",
                "vdw.01",
                "vdw.02",
                "vdw.03",
                "vdw.04",
                "vdw.05",
                "vdw.06",
                "vdw.07",
                "vdw.08",
                "vdw.09",
                "vdw.10",
                "vdw.11",
                "vdw.12",
                "vdw.13",
                "vdw.14",
                "vdw.15",
                "vdw.16",
                "vdw.17",
                "vdw.18",
                "vdw.19",
                "vdw.20",
                # "rest.00",
                # "rest.01",
                # "rest.02",
                # "rest.03",
                # "rest.04",
                # "rest.05",
                # "rest.06",
                # "rest.07",
                # "rest.08",
                # "rest.09",
                # "rest.10",
                # "rest.11",
            ],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        logging.info("Making universes for compound %s...", cmp)
        ensemble_handler.make_universes()
        # logging.info("Checking residue consistency for compound %s...", cmp)
        # check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # Featurize
        logging.info("Featurizing compound %s...", cmp)
        featurizer = BindingPocketVolumeFeaturizer(ensemble_handler)
        featurizer.featurize(
            selection="resid " + config["pocket_residues_string"],
            method="alpha",
            alpha=3.0,
            use_pp_transforms=False,
            pbc_corrections=False,
        )
        featurizer.save_features(cmp_output_dir, overwrite=True)
        featurizer.load_features(input_dir=cmp_output_dir)

        # Plot features
        # Load data
        df = pd.read_csv(cmp_output_dir + "/BindingPocketVolume_features.csv")
        out_file = os.path.join(
            cmp_output_dir, "BindingPocketVolume_features_timeseries.png"
        )
        # Plot
        plt.figure(figsize=(8, 5))
        for ensemble, group in df.groupby("ensemble"):
            plt.plot(
                group["time_ps"], group["pocket_volume"], label=ensemble, linewidth=0.5
            )

        plt.xlabel("Time (ps)")
        plt.ylabel("Pocket Volume (Å$^3$)")
        plt.title("Binding Pocket Volume Over Time by Ensemble")
        plt.legend(title="Ensemble", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save or show
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to {out_file}")

        # Subset to only vdw ensembles from 10 to 20
        avg_df = (df.groupby("ensemble", as_index=False)["pocket_volume"].mean().sort_values("ensemble"))
        # Split ensemble column to get window and van
        def parse_ensemble(name):
            if "nvt" in name:
                return "nvt", "nvt"
            elif "apo" in name:
                return "apo", "apo"
            elif "vdw." in name:
                parts = name.split("_")
                van = parts[1]+'_'+parts[2]  # e.g., "van_1"
                window = name.split("vdw.")[-1]  # e.g., "00"
                return van, f"vdw.{window}"
            else:
                # it's a van_X holo run
                van = [p for p in name.split("_") if p.startswith("van")][0]
                return van, "holo"

        # Apply parsing
        avg_df[["van", "window"]] = avg_df["ensemble"].apply(
            lambda x: pd.Series(parse_ensemble(x))
        )

        # Save to CSV
        avg_df.to_csv(os.path.join(cmp_output_dir, "BindingPocketVolume_features_avg.csv"))

        # Plot line plot of average pocket volume vs window colored by van
        plt.figure(figsize=(10, 6))
        for van, group in avg_df.groupby("van"):
            plt.plot(
                group["window"],
                group["pocket_volume"],
                marker="o",
                label=van,
            )
        plt.xlabel("Window")
        plt.ylabel("Average Pocket Volume (Å$^3$)")
        plt.title("Average Binding Pocket Volume by Window and Vanilla")
        plt.legend(title="Vanilla", bbox_to_anchor=(1.05, 1),
                     loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(cmp_output_dir, "BindingPocketVolume_features_avg.png"), dpi=300
        )
        plt.close()

        # logging.info("Performing dimensionality reduction for compound %s...", cmp)
        # dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
        # dimreducer.reduce_dimensions()
        # dimreducer.calculate_projections()
        # dimreducer.save_projection_df(
        #     save_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
        # )

        # # Projection df without nvt:
        # projection_df = dimreducer.get_pca_projection_df()
        # # Prepare entropy heatmap
        # projection_df = projection_df[~projection_df["ensemble"].str.contains("nvt")]
        # plot_entropy_heatmaps(
        #     cmp=cmp,
        #     entropy_metric="jsd",
        #     columns_to_consider=["PC1", "PC2"],
        #     ensemble_handler=ensemble_handler,
        #     projection_df=projection_df,
        #     output_dir=cmp_output_dir,
        # )

        # # Visualization
        # projection_df = dimreducer.get_pca_projection_df()
        # # remove rows with ensemble containing nvt
        # projection_df = projection_df[~projection_df["ensemble"].str.contains("apo")]

        # logging.info("Visualizing compound %s...", cmp)
        # dimred_visualizer = DimRedVisualizer(
        #     projection_df=projection_df, data_name="PCA"
        # )
        # dimred_visualizer.plot_dimred_sims(
        #     save_path=os.path.join(cmp_output_dir, "pca_components_ensemble_noapo.png"),
        #     highlights=[f"{cmp}_nvt"],
        # )
        # dimred_visualizer.plot_dimred_time(
        #     save_path=os.path.join(cmp_output_dir, "pca_components_time_noapo.png")
        # )

        # # Visualization with apo
        # projection_df = dimreducer.get_pca_projection_df()

        # dimred_visualizer = DimRedVisualizer(
        #     projection_df=projection_df, data_name="PCA"
        # )
        # dimred_visualizer.plot_dimred_sims(
        #     save_path=os.path.join(cmp_output_dir, "pca_components_ensemble.png"),
        #     highlights=[f"{cmp}_nvt"],
        # )
        # dimred_visualizer.plot_dimred_time(
        #     save_path=os.path.join(cmp_output_dir, "pca_components_time.png")
        # )
        # plot_eigenvalues(
        #     pca_object=dimreducer.get_pca(),
        #     n_components=8,
        #     save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        # )
        # plot_pca_components(
        #     pca_object=dimreducer.get_pca(),
        #     feature_df=featurizer.get_feature_df(),
        #     num=8,
        #     save_path=os.path.join(cmp_output_dir, "pca_components.png"),
        # )


if __name__ == "__main__":
    main()
