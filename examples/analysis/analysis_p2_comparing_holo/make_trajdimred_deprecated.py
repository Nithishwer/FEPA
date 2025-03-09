# Add a this is deprecated warning
import warnings
warnings.warn("This script is deprecated and will be removed in future versions.", DeprecationWarning)


import os
import sys
import logging
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from typing import Optional, Type
from core.DimReducer import PCAReducer
from core.DimReducer import UMAPReducer
from core.DimReducer import DimReducer
from core.TrajectoryLoader import TrajectoryLoader
from core.Visualizer import Visualizer
from core.JSAnalyzer import JSAnalyzer
from utils.file_utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def load_trajectories_for_compound(config: dict, cmp: str) -> TrajectoryLoader:
    """Loads trajectories for a given compound."""
    traj_loader = TrajectoryLoader(vanilla_pdb_name_template='OX2R_{CMP_NAME}_van_{VAN_REP_NO}_bp_protein.pdb',
                                   vanilla_xtc_name_template='OX2R_{CMP_NAME}_van_{VAN_REP_NO}_bp_protein_all.xtc',
                                   nvt_pdb_name_template='OX2R_{CMP_NAME}_nvt_bp_protein.pdb',
                                   nvt_xtc_name_template='OX2R_{CMP_NAME}_nvt_bp_protein_all.xtc',
                                   apo_pdb_name_template='OX2R_apo_{VAN_REP_NO}_bp_protein.pdb',
                                   apo_xtc_name_template='OX2R_apo_{VAN_REP_NO}_bp_protein_all.xtc')
    for van_rep_no in [1, 2, 3]:
        traj_loader.append_vanilla_path(compound_name=cmp, van_rep_no=van_rep_no)
        traj_loader.append_apo_path(van_rep_no=van_rep_no)
        traj_loader.append_nvt_path(compound_name=cmp)
        logging.info(f"Loaded trajectories for {cmp} and rep {van_rep_no}.")
    return traj_loader

def analyze_with_dim_reduction(traj_loader: TrajectoryLoader, cmp_output_dir: str, reducer_class: Type[DimReducer], reducer_name: str,compound_name: str) -> None:
    """Performs analysis with a given dimensionality reduction technique."""
    u, metadata_df = traj_loader.combine_trajectories()
    reducer_engine = reducer_class(u, metadata_df) 
    reducer_engine.analyse_sda()
    reducer_engine.reduce_dimensions(n_components=4)
    reducer_engine.cluster() 
    output_df = reducer_engine.get_output()
    output_df.to_csv(os.path.join(cmp_output_dir, f'{reducer_name}_df.csv'))
    reducer_class.save_object(reducer_engine, os.path.join(cmp_output_dir, f'{reducer_name}_engine.pkl'), output_dir='.') # Use class method for saving
    Plotting_engine = Visualizer(dimred_df_w_metadata=output_df, data_name=reducer_name.upper(), compound_name=compound_name)
    Plotting_engine.plot_dimred_sims(save_path=os.path.join(cmp_output_dir, f'{reducer_name}_dimred_sims.png'))
    Plotting_engine.plot_dimred_time(save_path=os.path.join(cmp_output_dir, f'{reducer_name}_dimred_time.png'))
    Plotting_engine.plot_dimred_cluster(save_path=os.path.join(cmp_output_dir, f'{reducer_name}_dimred_cluster.png'), cluster_column='cluster')

    if reducer_name == 'PCA': # Elbow plot is PCA-specific, keep it conditional if needed
        reducer_engine.elbow_plot(save_path=os.path.join(cmp_output_dir, 'pca_elbow_plot.png'))

def main():

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = '../../../data/processed/'
    analysis_output_dir = os.path.join(parent_output_dir, 'analysis_p2_comparing_holo')
 
    for cmp in config["compounds"][:]:

        # Log
        logging.info(f"Analyzing compound {cmp}...")

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Load trajectories
        traj_loader = load_trajectories_for_compound(config, cmp)

        # Perform analysis with PCA
        analyze_with_dim_reduction(traj_loader, cmp_output_dir, PCAReducer, 'PCA', compound_name=cmp)

        # Perform analysis with UMAP
        analyze_with_dim_reduction(traj_loader, cmp_output_dir, UMAPReducer, 'UMAP', compound_name=cmp)

if __name__ == '__main__':
    main()
