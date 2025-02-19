import json
import os
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.file_utils import load_config
from utils.md_utils import annotate_binding_pocket, save_universes
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Main function to extract binding pocket trajectories based on config."""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.json")
    config = load_config(config_path)

    # Ensure output directory exists
    output_dir = '../../data/raw/'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize other variables
    universe_dict = {}
    bp_string = f"protein and resid {config['pocket_residues_string']}"

    for cmp in config["compounds"]:
        for rep_no in [1, 2, 3]:
            vanilla_compound_path = config['vanilla_path_template'].format(CMP_NAME=cmp, REP_NO=rep_no) if rep_no > 1 else config['vanilla_path_old'].format(CMP_NAME=cmp)
            gro_file = os.path.join(vanilla_compound_path, "npt.gro")
            xtc_file = os.path.join(vanilla_compound_path, "prod.xtc")
            nvt_gro_file = os.path.join(vanilla_compound_path, "nvt.gro")
            nvt_xtc_file = os.path.join(vanilla_compound_path, "nvt.xtc")
            universe_dict[f'OX2R_{cmp}_van_{rep_no}'] = annotate_binding_pocket(gro_file,xtc_file, bp_string)
            universe_dict[f'OX2R_{cmp}_nvt'] = annotate_binding_pocket(nvt_gro_file, nvt_xtc_file, bp_string)

    for rep_no in [1, 2, 3]:
        apo_compound_path = config['apo_path_template'].format(REP_NO=rep_no)
        gro_file = os.path.join(apo_compound_path, "npt.gro")
        xtc_file = os.path.join(apo_compound_path, "prod.xtc")
        universe_dict[f'OX2R_apo_{rep_no}'] = annotate_binding_pocket(gro_file, xtc_file, bp_string)

    # Saving universes
    save_universes(universe_dict=universe_dict, selection_string='protein', save_location=output_dir)

if __name__ == "__main__":
    main()
