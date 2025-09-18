import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_apo, load_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("a10_sidechain_torsions_analysis", "wdir")

    for cmp in config["compounds"]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)

        # path_dict = load_paths_for_apo(
        #     config,
        #     bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket 
        #     )

        path_dict = load_paths_for_compound(
            config,
            cmp,
            bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246", # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket
            apo = True
        )


        # Load trajectories
        # logging.info("Loading trajectories for compound %s ...", cmp)
        # ensemble_handler = EnsembleHandler(path_dict)
        # ensemble_handler.make_universes()

        # # Featurize
        # logging.info("Featurizing binding pocket waters ...")
        # sc_tors_featurizer = SideChainTorsionsFeaturizer(
        #     ensemble_handler=ensemble_handler)

        # sc_tors_featurizer.featurize()

        # # Save features
        # logging.info("Saving features for compound %s ...", cmp)
        # sc_tors_featurizer.save_features(cmp_output_dir, overwrite=True)

        features_df = pd.read_csv(os.path.join(cmp_output_dir, "SideChainTorsions_features.csv"))

        # Subset features df columns to only ones that have binding pocket resids
        features_df_columns = features_df.columns
        # logging.info("Features df columns: %s", features_df_columns)
        binding_pocket_resids = ['57','58','61','64','83','84','87','88','91','92','173','177','218','221','235','238','239','242','243','246']
        # logging.info("Binding pocket resids: %s", binding_pocket_resids)
        features_df_columns_bp = [col for col in features_df_columns if any(resid in col for resid in binding_pocket_resids)] + ['timestep','ensemble']
        # logging.info("Binding pocket features df columns: %s", features_df_columns_bp)
        features_df = features_df[features_df_columns_bp]

        # Make new features df for components
        features_df_w_components = features_df.copy()

        # Identify CHI columns
        chi_cols = [col for col in features_df_w_components.columns if 'CHI' in col]

        # Create sine and cosine transformed columns
        for col in chi_cols:
            features_df_w_components[f'{col}_sin'] = np.sin(np.radians(features_df_w_components[col]))
            features_df_w_components[f'{col}_cos'] = np.cos(np.radians(features_df_w_components[col]))

        # Optionally drop original CHI columns
        features_df_w_components.drop(columns=chi_cols, inplace=True)

        # Display the transformed DataFrame
        print(features_df_w_components.head())

        # Save features with components
        features_df_w_components.to_csv(os.path.join(cmp_output_dir, "SideChainTorsions_features_with_components.csv"), index=False)


        logging.info("Performing dimensionality reduction for compound %s...", cmp)
        dimreducer = PCADimReducer(
            features_df_w_components, n_components=8
        )
        dimreducer.reduce_dimensions(feature_column_keyword="CHI")
        dimreducer.calculate_projections()
        dimreducer.save_projection_df(
            save_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
        )

        # Projection df without nvt:
        projection_df = dimreducer.get_pca_projection_df()


        # Dimensionality Reduction
        projection_df = projection_df[~projection_df["ensemble"].str.contains("nvt")]
        plot_entropy_heatmaps(
            cmp=cmp,
            entropy_metric="jsd",
            columns_to_consider=["PC1", "PC2"],
            projection_df=projection_df,
            output_dir=cmp_output_dir,
        )

        # Visualization without apo
        projection_df = dimreducer.get_pca_projection_df()
        projection_df = projection_df[~projection_df["ensemble"].str.contains("apo")]

        logging.info("Visualizing compound %s...", cmp)
        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(cmp_output_dir, "pc1_pc2_scatter_ensemble_noapo.png"),
            highlights=[f"{cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "pc1_pc2_scatter_time_noapo.png")
        )

        # Visualization with apo
        projection_df = dimreducer.get_pca_projection_df()

        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(cmp_output_dir, "pc1_pc2_scatter_ensemble.png"),
            highlights=[f"{cmp}_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "pc1_pc2_scatter_time.png")
        )
        plot_eigenvalues(
            pca_object=dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        )
        # Plot PCA Components for apo only
        plot_pca_components(
            pca_object=dimreducer.get_pca(),
            feature_df=features_df_w_components[features_df_w_components["ensemble"].str.contains("apo")],  
            num=8,
            save_path=os.path.join(cmp_output_dir, "pca_components_apo.png"),
            feature_column_keyword="CHI",
        ) 
        # Plot PCA components for vanilla only
        plot_pca_components(
            pca_object=dimreducer.get_pca(),
            feature_df=features_df_w_components[features_df_w_components["ensemble"].str.contains("van")],
            num=8,
            save_path=os.path.join(cmp_output_dir, "pca_components_vanilla.png"),
            feature_column_keyword="CHI",
        )
  
if __name__ == "__main__":
    main()
