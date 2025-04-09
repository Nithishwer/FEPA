"""
This module contains the EnsembleHandler class, used to load and handle MD ensembles.
"""

import warnings
from typing import Dict
import MDAnalysis as mda
import MDAnalysis.transformations as trans
import logging

warnings.filterwarnings("ignore")  # Suppress MDAnalysis warnings


class EnsembleHandler:
    """
    Class to handle MD ensembles.

    TODO: Change to use only path data and not store universes in memory.
    """

    def __init__(self, path_dict, data_dir: str = "../../data/trajectories/") -> None:
        """
        Initializes EnsembleLoader with data directory.

        Args:
            data_dir (str, optional): Path to the raw data directory, ('../../data/trajectories/')
        """
        self.data_dir = data_dir
        self.path_dict = path_dict
        self.universe_dict = {}

    def make_universes(self):
        """
        Makes and stores MDAnalysis universes from the path dictionary.
        """
        for key in self.path_dict.keys():
            tpr_path = self.path_dict[key]["tpr"]
            xtc_path = self.path_dict[key]["xtc"]
            bp_selection_string = self.path_dict[key]["bp_selection_string"]
            self.universe_dict[key] = mda.Universe(tpr_path, xtc_path)
            bp_atoms = self.universe_dict[key].select_atoms(bp_selection_string)
            bp_atoms.residues.segments = self.universe_dict[key].add_Segment(segid="BP")
            non_bp_atoms = self.universe_dict[key].select_atoms("not segid BP")
            non_bp_atoms.residues.segments = self.universe_dict[key].add_Segment(
                segid="NOBP"
            )

    def get_timestep_from_universe(self, key: str) -> list:
        """
        Extracts the timestep data from a universe.

        Args:
            key (str): Key for the universe in the universe dictionary.

        Returns:
            list
        """
        timestep_data = []
        for ts in self.universe_dict[key].trajectory:
            timestep_data.append(ts.time)
        return timestep_data

    def get_universe_dict(self) -> Dict[str, mda.Universe]:
        """
        Returns the universe dictionary.

        Returns:
            Dict[str, mda.Universe]
        """
        return self.universe_dict

    def get_path_dict(self) -> Dict[str, Dict[str, str]]:
        """
        Returns the path dictionary.

        Returns:
            Dict[str, Dict[str, str]]
        """
        return self.path_dict

    def dump_gro(self, ensemble: str, timestep: float, save_path: str) -> None:
        """
        Dumps the universe to a GRO file.

        Args:
            key (str): Key for the universe in the universe dictionary.
            output_path (str): Path to the output GRO file.
        """
        for ts in self.universe_dict[ensemble].trajectory:
            if ts.time == timestep:
                self.universe_dict[ensemble].trajectory[ts.frame]
                break
        logging.info(
            f"Dumping GRO file for {ensemble} at timestep {self.universe_dict[ensemble].trajectory.frame} to {save_path}"
        )
        with mda.Writer(
            save_path, n_atoms=self.universe_dict[ensemble].atoms.n_atoms
        ) as writer:
            writer.write(self.universe_dict[ensemble].atoms)

    def make_ensemble_pp_trans(self, ensemble: str) -> list:
        """
        Make transformations for unwrapping the trajectory, centering
        the protein and wrapping the rest with whole fragments with
        the first ensemble
        """
        tpr_path = self.path_dict[ensemble]["tpr"]
        xtc_path = self.path_dict[ensemble]["xtc"]

        # Load trajectory
        u = mda.Universe(tpr_path, xtc_path)
        protein = u.select_atoms("protein")
        not_protein = u.select_atoms("not protein")

        # Make list of transformations
        transformations = [
            trans.unwrap(protein),
            trans.center_in_box(protein, wrap=True),
            trans.wrap(not_protein),
        ]

        return transformations
