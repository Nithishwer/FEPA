import os
import json
import types
from typing import Optional, Type
import MDAnalysis as mda

def load_paths_for_compound(config: dict, cmp: str, bp_selection_string: str) -> dict:
    """Loads MD trajectory paths for a given compound."""
    path_dict = {}
    for van_rep_no in [1, 2, 3]:
        path_dict[f'{cmp}_van_{van_rep_no}'] = {
            'pdb': os.path.join(config['vanilla_path_template_old' if van_rep_no == 1 else 'vanilla_path_template'].format(CMP_NAME=cmp, REP_NO=van_rep_no), 'npt.gro'), 
            'xtc': os.path.join(config['vanilla_path_template_old' if van_rep_no == 1 else 'vanilla_path_template'].format(CMP_NAME=cmp, REP_NO=van_rep_no), 'prod.xtc'),
            'bp_selection_string': bp_selection_string
        }
        path_dict[f'{cmp}_nvt'] = {
            'pdb': os.path.join(config['vanilla_path_template_old' if van_rep_no ==1 else 'vanilla_path_template'].format(CMP_NAME=cmp, REP_NO=van_rep_no), 'nvt.gro'), 
            'xtc': os.path.join(config['vanilla_path_template_old' if van_rep_no ==1 else 'vanilla_path_template'].format(CMP_NAME=cmp, REP_NO=van_rep_no), 'nvt.xtc'),
            'bp_selection_string': bp_selection_string
        }
    for van_rep_no in [1, 2, 3]:
        path_dict[f'apo_{van_rep_no}'] = {
            'pdb': os.path.join(config['apo_path_template'].format(REP_NO=van_rep_no), 'npt.gro'), 
            'xtc': os.path.join(config['apo_path_template'].format(REP_NO=van_rep_no), 'prod.xtc'),
            'bp_selection_string': bp_selection_string
        }
    return path_dict