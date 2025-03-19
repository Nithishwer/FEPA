"""This script compares the overlap in the different holo states of the complexes"""

import os
import logging
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    compute_histograms,
    plot_jsd_histograms,
    sort_ensemble_by_pc,
    plot_eigenvalues,
    plot_pca_components,
)
from fepa.core.analyzers import compute_relative_entropy

from fepa.utils.md_utils import (
    check_bp_residue_consistency,
    get_ca_rmsf,
    plot_ca_rmsfs,
    generate_restraint_dicts,
    generate_gmx_restraints_file,
)
from fepa.utils.dimred_utils import elbow_plot_pca, cluster_pca
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_paths_for_compound

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = "../../wdir/data/"
    analysis_output_dir = os.path.join(parent_output_dir, "analysis_p2_comparing_holo")

    for cmp in config["compounds"][:1]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)
        path_dict = load_paths_for_compound(
            config,
            cmp,
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        )

        # Testing with only few trajectories
        # path_dict = {k: v for k, v in path_dict.items() if "apo" in k}

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        logging.info("Making universes for compound %s...", cmp)
        ensemble_handler.make_universes()
        logging.info("Checking residue consistency for compound %s...", cmp)
        check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # Featurize
        logging.info("Featurizing compound %s...", cmp)
        featurizer = SelfDistanceFeaturizer(ensemble_handler)
        featurizer.featurize()
        featurizer.save_features(output_dir=cmp_output_dir, overwrite=True)
        featurizer.load_features(input_dir=cmp_output_dir)

        # Dimensionality Reduction
        logging.info("Performing dimensionality reduction for compound %s...", cmp)
        dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
        dimreducer.reduce_dimensions()
        dimreducer.calculate_projections()
        dimreducer.save_projection_df(
            save_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
        )
        dimreducer.load_projection_df(
            load_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
        )
        elbow_plot_pca(
            pca_projection_df=dimreducer.get_pca_projection_df(),
            save_path=os.path.join(cmp_output_dir, "pca_elbow_plot.png"),
        )
        projection_df_w_clusters = cluster_pca(
            pca_projection_df=dimreducer.get_pca_projection_df(), n_clusters=5
        )

        # Visualization
        logging.info("Visualizing compound %s...", cmp)
        dimred_visualizer = DimRedVisualizer(
            projection_df=projection_df_w_clusters, data_name="PCA"
        )
        dimred_visualizer.plot_dimred_sims(
            save_path=os.path.join(cmp_output_dir, "pca_components_ensemble.png"),
            highlights=["42922_nvt"],
        )
        dimred_visualizer.plot_dimred_time(
            save_path=os.path.join(cmp_output_dir, "pca_components_time.png")
        )
        dimred_visualizer.plot_dimred_cluster(
            save_path=os.path.join(cmp_output_dir, "pca_components_cluster.png"),
            cluster_column="cluster",
        )
        plot_eigenvalues(
            pca_object=dimreducer.get_pca(),
            n_components=8,
            save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
        )
        plot_pca_components(
            pca_object=dimreducer.get_pca(),
            feature_df=featurizer.get_feature_df(),
            num=8,
            save_path=os.path.join(cmp_output_dir, "pca_components.png"),
        )

        # Save sorted ensemble
        sort_ensemble_by_pc(
            pca=dimreducer.get_pca(),
            feature_df=featurizer.get_feature_df(),
            ensemble_handler=ensemble_handler,
            save_path=os.path.join(cmp_output_dir, "sorted_apo3_PC1.xtc"),
            ensemble="apo_3",
            projection_num=1,
        )
        sort_ensemble_by_pc(
            pca=dimreducer.get_pca(),
            feature_df=featurizer.get_feature_df(),
            ensemble_handler=ensemble_handler,
            save_path=os.path.join(cmp_output_dir, "sorted_apo3_PC2.xtc"),
            ensemble="apo_3",
            projection_num=2,
        )

        # JS Analysis
        logging.info("Performing Jensen-Shannon analysis for compound %s...", cmp)
        relative_entropy_dict = compute_relative_entropy(
            feature_df=featurizer.get_feature_df(),
            ensemble1="42922_van_1",
            ensemble2="apo_1",
            num_bins=50,
        )
        histograms = compute_histograms(
            feature_df=featurizer.get_feature_df(),
            key="ensemble",
            value_1="42922_van_1",
            value_2="apo_1",
            num_bins=50,
        )
        plot_jsd_histograms(
            relative_entropy_dict=relative_entropy_dict,
            histograms=histograms,
            save_path=os.path.join(cmp_output_dir, "jsd_histograms.png"),
        )

        # RMSF
        rmsf_dict_1 = get_ca_rmsf(
            pdb_path=path_dict["42922_van_1"]["pdb"],
            xtc_path=path_dict["42922_van_1"]["xtc"],
        )
        rmsf_dict_2 = get_ca_rmsf(
            pdb_path=path_dict["apo_1"]["pdb"], xtc_path=path_dict["apo_1"]["xtc"]
        )
        pocket_resids = [int(i) for i in config["pocket_residues_string"].split("  ")]
        plot_ca_rmsfs(
            rmsf_dict_list=[rmsf_dict_1, rmsf_dict_2],
            labels=["apo_1", "42922_van_1"],
            pocket_residues=pocket_resids,
            save_path=os.path.join(cmp_output_dir, "ca_rmsf.png"),
        )

        # Select restraints with RMSF
        restr_1_dict, restr_2_dict = generate_restraint_dicts(
            relative_entropy_dict, rmsf_dict_1, rmsf_dict_2, scaling_factor=1.0
        )

        # Visualize restraints
        plot_jsd_histograms(
            relative_entropy_dict=relative_entropy_dict,
            histograms=histograms,
            save_path=os.path.join(cmp_output_dir, "jsd_histograms.png"),
            restr_u1_dict=restr_1_dict,
            restr_u2_dict=restr_2_dict,
        )

        # Write restraints to file
        generate_gmx_restraints_file(
            ensemble_handler.get_universe_dict()["42922_van_1"],
            rmsf_dict_1,
            a=1,
            output_file="ca_restraints.itp",
        )


if __name__ == "__main__":
    main()
