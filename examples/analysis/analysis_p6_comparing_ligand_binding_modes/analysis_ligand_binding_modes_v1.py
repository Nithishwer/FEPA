"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
import gromacs

from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
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
from fepa.utils.md_utils import write_traj_without_PBC_jumps

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
    analysis_output_dir = os.path.join(
        parent_output_dir, "analysis_p6_comparing_ligand_binding_modes"
    )
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Prepare paths
    path_dict = load_paths_for_compound(
        config,
        cmp="42922",
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    )

    logging.info("Path dict: %s", path_dict["42922_van_1"])

    gro_path = path_dict["42922_van_1"]["pdb"]
    xtc_path = path_dict["42922_van_1"]["xtc"]
    tpr_file = path_dict["42922_van_1"]["pdb"].replace(".gro", ".tpr")

    # Make tpr from gro and xtc
    gromacs.grompp(
        f="/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/OX2_42922/vanilla/minim.mdp",
        c=gro_path,
        p="/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/OX2_42922/vanilla/topol.top",
        o=tpr_file,
    )

    write_traj_without_PBC_jumps(
        gro_path=tpr_file,  # should take tpr
        xtc_path=xtc_path,
        output_gro_path=os.path.join(analysis_output_dir, "prod_no_pbc.gro"),
        output_xtc_path=os.path.join(analysis_output_dir, "prod_no_pbc.xtc"),
        centering_selection_string="protein",
    )

    write_traj_without_PBC_jumps(
        gro_path=tpr_file,  # should take tpr
        xtc_path=xtc_path,
        output_gro_path=os.path.join(analysis_output_dir, "prod_no_pbc_nowater.gro"),
        output_xtc_path=os.path.join(analysis_output_dir, "prod_no_pbc_nowater.xtc"),
        centering_selection_string="protein",
        saving_selection_string="not resname SOL",
    )

    # This works -> Check how the output is. Is it reasonable?

    # Then run this for everything and then work towards getting rotatable bonds in the ligand

    # # Load trajectories
    # ensemble_handler = EnsembleHandler(path_dict)
    # ensemble_handler.make_universes()
    # check_bp_residue_consistency(ensemble_handler.get_universe_dict())

    # # Featurize
    # featurizer = SelfDistanceFeaturizer(ensemble_handler)
    # featurizer.featurize()
    # # featurizer.save_features(output_dir=analysis_output_dir, overwrite=True)

    # # Dimensionality Reduction
    # dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
    # dimreducer.reduce_dimensions()
    # dimreducer.calculate_projections()
    # dimreducer.save_projection_df(
    #     save_path=os.path.join(analysis_output_dir, "pca_projection_df.csv")
    # )

    # # Projection df without nvt:
    # projection_df = dimreducer.get_pca_projection_df()
    # plot_entropy_heatmaps(
    #     cmp="apo",  # Abusing function should make it more general
    #     entropy_metric="jsd",
    #     columns_to_consider=["PC1", "PC2"],
    #     ensemble_handler=ensemble_handler,
    #     projection_df=projection_df,
    #     output_dir=analysis_output_dir,
    # )
    # # Visualization
    # projection_df = dimreducer.get_pca_projection_df()
    # dimred_visualizer = DimRedVisualizer(projection_df=projection_df, data_name="PCA")
    # dimred_visualizer.plot_dimred_sims(
    #     save_path=os.path.join(analysis_output_dir, "pca_components_ensemble.png")
    # )
    # dimred_visualizer.plot_dimred_time(
    #     save_path=os.path.join(analysis_output_dir, "pca_components_time.png")
    # )
    # plot_eigenvalues(
    #     pca_object=dimreducer.get_pca(),
    #     n_components=8,
    #     save_path=os.path.join(analysis_output_dir, "eigenvalues.png"),
    # )
    # plot_pca_components(
    #     pca_object=dimreducer.get_pca(),
    #     feature_df=featurizer.get_feature_df(),
    #     num=8,
    #     save_path=os.path.join(analysis_output_dir, "pca_components.png"),
    # )


if __name__ == "__main__":
    main()
