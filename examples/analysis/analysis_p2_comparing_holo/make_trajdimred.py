import os
import sys
import logging
import pandas as pd
sys.path.append('/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/fepa/')
from typing import Optional, Type
from fepa.core.DimReducer import PCAReducer
from fepa.core.DimReducer import UMAPReducer
from fepa.core.DimReducer import DimReducer
from fepa.core.Visualizer import Visualizer
from fepa.core.JSAnalyzer import JSAnalyzer
from utils.file_utils import load_config
from utils.path_utils import load_paths_for_compound
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional



def analyze_with_dim_reduction(path_dict: dict, cmp_output_dir: str, reducer_class: Type[DimReducer], reducer_name: str,compound_name: str) -> None:
    """Performs analysis with a given dimensionality reduction technique."""
    reducer_engine = reducer_class(path_dict) 
    reducer_engine.make_universes()
    reducer_engine.make_sdas()
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
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.json")
    config = load_config(config_path)
    parent_output_dir = '../../wdir/data/processed/'
    analysis_output_dir = os.path.join(parent_output_dir, 'analysis_p2_comparing_holo')
 
    for cmp in config["compounds"][:]:

        # Log
        logging.info(f"Analyzing compound {cmp}...")

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Load trajectories
        path_dict = load_paths_for_compound(config, cmp, bp_selection_string='resid '+config['pocket_residues_string'])

        # Perform analysis with PCA
        analyze_with_dim_reduction(path_dict, cmp_output_dir, PCAReducer, 'PCA', compound_name=cmp)

        # Perform analysis with UMAP
        analyze_with_dim_reduction(path_dict, cmp_output_dir, UMAPReducer, 'UMAP', compound_name=cmp)

if __name__ == '__main__':
    main()
