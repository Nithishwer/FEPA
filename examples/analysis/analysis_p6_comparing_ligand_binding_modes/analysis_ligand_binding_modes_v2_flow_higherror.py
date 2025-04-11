"""This script compares the overlap in the different holo states of the complexes"""

import logging
import os
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.utils.md_utils import (
    check_bp_residue_consistency,
)
from fepa.flows.analysis_flows import torsions_analysis_workflow

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

    # Creating van_list and leg_window_list
    van_list = [i for i in range(1, 4)]
    leg_window_list = (
        [f"coul.{i:02}" for i in range(0, 11)]
        + [f"vdw.{i:02}" for i in range(0, 12)]
        + [f"rest.{i:02}" for i in range(0, 11)]
    )

    cmps_of_interest = ["48951", "47594", "49599", "52542", "47821"]

    for cmp in cmps_of_interest:
        # Prepare paths
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp=cmp,
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            van_list=van_list,
            leg_window_list=leg_window_list,
            apo=False,
        )
        # path_dict = load_paths_for_compound(
        #     config,
        #     cmp="42922",
        #     bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        #     apo=False,
        # )

        logging.info("Path dict: %s", path_dict)

        # Load trajectories
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()
        check_bp_residue_consistency(ensemble_handler.get_universe_dict())

        # Using the flow
        torsions_analyzer = torsions_analysis_workflow(
            ensemble_handler=ensemble_handler, sel="resname unk"
        )
        torsions_analyzer.featurize()
        torsions_analyzer.reduce_dimensions()
        torsions_analyzer.plot_dimred_sims(
            save_path=os.path.join(analysis_output_dir, f"dimred_sims_{cmp}.png")
        )
        torsions_analyzer.plot_dimred_time(
            save_path=os.path.join(analysis_output_dir, f"dimred_time_{cmp}.png")
        )
        torsions_analyzer.cluster_pca(n_clusters=4, n_components=1)
        torsions_analyzer.plot_dimred_cluster(
            save_path=os.path.join(analysis_output_dir, f"dimred_cluster_{cmp}.png"),
            cluster_column="cluster",
        )
        torsions_analyzer.plot_all_torsions_jsd_histograms(
            output_dir=analysis_output_dir,
        )
        torsions_analyzer.annotate_ensembles_in_dimred_df(
            save_path=os.path.join(analysis_output_dir, f"projection_df_{cmp}.csv")
        )
        torsions_analyzer.plot_cluster_distr(
            save_path=os.path.join(analysis_output_dir, f"cluster_distr_{cmp}.png"),
            groupby=["sim_type", "cluster"],
        )


if __name__ == "__main__":
    main()
