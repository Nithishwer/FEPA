import os
import sys
import warnings
# Deprecation
warnings.warn("This script is deprecated and will be removed in future versions.", DeprecationWarning)
import logging
import pandas as pd
import MDAnalysis as mda
from typing import List, Tuple, Dict, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.check_consistency import check_consistency

warnings.filterwarnings('ignore')  # Suppress MDAnalysis warnings

class TrajectoryLoader():
    """
    Loads and combines MD trajectories from specified file paths, ensuring consistency
    and creating associated metadata.

    This class is designed to handle 'vanilla' and 'apo' type trajectories based on
    provided file name templates and a raw data directory. It checks for PDB consistency
    before combining trajectories into a single MDAnalysis Universe and generates a metadata
    DataFrame.
    """
    def __init__(self, vanilla_pdb_name_template: str, vanilla_xtc_name_template: str,
                 apo_pdb_name_template: str, apo_xtc_name_template: str,
                 nvt_pdb_name_template: str, nvt_xtc_name_template: str,
                 raw_data_dir: str = '../../../data/raw/') -> None:
        """
        Initializes TrajectoryLoader with file name templates and data directory.

        Args:
            vanilla_pdb_name_template (str): Template string for vanilla PDB file names, e.g., "{CMP_NAME}_van_{VAN_REP_NO}.pdb".
            vanilla_xtc_name_template (str): Template string for vanilla XTC file names, e.g., "{CMP_NAME}_van_{VAN_REP_NO}.xtc".
            apo_pdb_name_template (str): Template string for apo PDB file names, e.g., "apo_{VAN_REP_NO}.pdb".
            apo_xtc_name_template (str): Template string for apo XTC file names, e.g., "apo_{VAN_REP_NO}.xtc".
            raw_data_dir (str, optional): Path to the raw data directory. Defaults to '../../../data/raw/'.
        """
        self.vanilla_xtc_name_template = vanilla_xtc_name_template
        self.vanilla_pdb_name_template = vanilla_pdb_name_template
        self.nvt_xtc_name_template = nvt_xtc_name_template
        self.nvt_pdb_name_template = nvt_pdb_name_template
        self.apo_xtc_name_template = apo_xtc_name_template
        self.apo_pdb_name_template = apo_pdb_name_template
        self.raw_data_dir = raw_data_dir
        self.pdb_path_list: List[str] = []
        self.xtc_path_list: List[str] = []
        self.id_list: List[str] = []
        self.metadata_df: Optional[pd.DataFrame] = None

    def append_vanilla_path(self, compound_name: str, van_rep_no: int) -> None:
        """
        Appends file paths for a vanilla trajectory to the loader lists.

        Args:
            compound_name (str): Name of the compound.
            van_rep_no (int): Replica number for the vanilla simulation.
        """
        id = f'{compound_name}_van_{van_rep_no}'
        pdb_file_path = os.path.join(self.raw_data_dir, self.vanilla_pdb_name_template.format(CMP_NAME=compound_name, VAN_REP_NO=van_rep_no))
        xtc_file_path = os.path.join(self.raw_data_dir, self.vanilla_xtc_name_template.format(CMP_NAME=compound_name, VAN_REP_NO=van_rep_no))
        self.pdb_path_list.append(pdb_file_path)
        self.xtc_path_list.append(xtc_file_path)
        self.id_list.append(id)
    
    def append_nvt_path(self, compound_name: str) -> None:
        """
        Appends file paths for an NVT trajectory to the loader lists.

        Args:
            compound_name (str): Name of the compound.
        """
        id = f'{compound_name}_nvt'
        pdb_file_path = os.path.join(self.raw_data_dir, self.nvt_pdb_name_template.format(CMP_NAME=compound_name))
        xtc_file_path = os.path.join(self.raw_data_dir, self.nvt_xtc_name_template.format(CMP_NAME=compound_name))
        self.pdb_path_list.append(pdb_file_path)
        self.xtc_path_list.append(xtc_file_path)
        self.id_list.append(id)

    def append_apo_path(self, van_rep_no: int) -> None:
        """
        Appends file paths for an apo trajectory to the loader lists.

        Args:
            van_rep_no (int): Replica number for the apo simulation.
        """
        id = f'apo_{van_rep_no}'
        pdb_file_path = os.path.join(self.raw_data_dir, self.apo_pdb_name_template.format(VAN_REP_NO=van_rep_no))
        xtc_file_path = os.path.join(self.raw_data_dir, self.apo_xtc_name_template.format(VAN_REP_NO=van_rep_no))
        self.pdb_path_list.append(pdb_file_path)
        self.xtc_path_list.append(xtc_file_path)
        self.id_list.append(id)

    def check_pdb_consistency(self) -> bool:
        """
        Checks if the PDB files in the pdb_path_list are consistent in terms of
        atom and residue names (using the external check_consistency function).

        Returns:
            bool: True if PDB files are consistent, False otherwise.
        """
        return check_consistency(self.pdb_path_list)

    def combine_trajectories(self) -> Tuple[Optional[mda.Universe], Optional[pd.DataFrame]]:
        """
        Combines the loaded trajectories into a single MDAnalysis Universe object
        if PDB files are consistent.

        Returns:
            Tuple[Optional[mda.Universe], Optional[pd.DataFrame]]: A tuple containing:
                - combined_universe (MDAnalysis.Universe or None): The combined Universe if successful, None otherwise.
                - metadata_df (pd.DataFrame or None): Metadata DataFrame if successful, None otherwise.
        """
        if self.check_pdb_consistency():
            reference_pdb = self.pdb_path_list[0]
            u = mda.Universe(reference_pdb, *self.xtc_path_list)
            self.make_metadata()
            logging.info("Trajectories combined successfully!")
            return u, self.metadata_df
        else:
            logging.error("PDB files are inconsistent. Trajectories not combined.")
            return None, None # Return None for both in case of failure

    def make_metadata(self) -> pd.DataFrame:
        """
        Generates a metadata DataFrame for the loaded trajectories.

        The DataFrame includes 'source_file', 'timestep', and 'id' columns.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        metadata = []
        for pdb, xtc, id in zip(self.pdb_path_list, self.xtc_path_list, self.id_list):
            u = mda.Universe(pdb, xtc)
            for ts in u.trajectory:
                metadata.append({
                    'source_file': xtc,
                    'timestep': ts.time,
                    'id': id
                })
        self.metadata_df = pd.DataFrame(metadata)
        return self.metadata_df