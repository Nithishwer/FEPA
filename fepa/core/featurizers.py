"""
This module contains classes for featurizing molecular dynamics trajectories of multiple ensembles
"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import multiprocessing
import numpy as np
from MDAnalysis.topology.guessers import guess_atom_element
from multiprocessing import Pool
from functools import partial
import MDAnalysis.analysis.hbonds
from MDAnalysis.analysis import align, rms
from pensa.features import (
    read_h_bonds,
    read_protein_sidechain_torsions,
)
from fepa.utils.featurize_utils import (
    compute_self_distances_with_transforms,
)
from fepa.utils.BAT_utils import read_BAT
from fepa.utils.water_utils import WaterOccupancyAnalysis
from fepa.core.ensemble_handler import EnsembleHandler
from MDAnalysis import transformations as trans


class BaseFeaturizer(ABC):
    """Base class for featurizers"""

    def __init__(self, ensemble_handler):
        self.ensemble_handler = ensemble_handler
        self.feature_df = None
        self.feature_type = "Generic"

    @abstractmethod
    def featurize(self):
        """Method to be implemented for extracting features."""

    def save_features(self, output_dir: str, overwrite: Optional[bool] = False):
        """Save features to a csv file"""
        out_file = os.path.join(output_dir, f"{self.feature_type}_features.csv")
        if not os.path.exists(output_dir):
            logging.info("Creating output directory %s", output_dir)
            os.makedirs(output_dir)
        elif os.path.exists(out_file):
            logging.info(
                "Output directory %s already exists. Overwrite set to %s",
                output_dir,
                overwrite,
            )
            if not overwrite:
                raise ValueError(f"Output directory {output_dir} already exists.")

        self.feature_df.to_csv(
            os.path.join(output_dir, f"{self.feature_type}_features.csv"), index=False
        )

    def load_features(self, input_dir: str):
        """Load features from a csv file"""
        csv_file = os.path.join(input_dir, f"{self.feature_type}_features.csv")
        self.feature_df = pd.read_csv(csv_file)
        logging.info("Loaded features from %s", csv_file)

    def get_feature_df(self) -> Dict[str, Any]:
        """Return the feature dataframe"""
        return self.feature_df


class SelfDistanceFeaturizer(BaseFeaturizer):
    """Class for featurizing self distances of atoms"""

    def __init__(self, ensemble_handler: EnsembleHandler):
        super().__init__(ensemble_handler)
        self.feature_type = "SelfDistance"

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr_path = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc_path = self.ensemble_handler.path_dict[ensemble]["xtc"]
            pp_trans = self.ensemble_handler.make_ensemble_pp_trans(ensemble)

            bp_selection_string = self.ensemble_handler.path_dict[ensemble][
                "bp_selection_string"
            ]
            name, data = compute_self_distances_with_transforms(
                tpr_path=tpr_path,
                xtc_path=xtc_path,
                selection=bp_selection_string,
                start=None,
                stop=None,
                step=1,
                transformations=pp_trans,
                pbc=True,
                feature_prefix="DIST",
            )
            ensemble_feature_df = pd.DataFrame(data, columns=name)
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class BATFeaturizer(BaseFeaturizer):
    """Class for featurizing torsions of selected atoms"""

    def __init__(self, ensemble_handler: EnsembleHandler, sel: str):
        super().__init__(ensemble_handler)
        self.feature_type = "BAT"
        self.selection_string = sel

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr_path = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc_path = self.ensemble_handler.path_dict[ensemble]["xtc"]
            name, data = read_BAT(
                tpr_path,
                xtc_path,
                sel=self.selection_string,
            )
            ensemble_feature_df = pd.DataFrame(data, columns=name)
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class TorsionsFeaturizer(BaseFeaturizer):
    """Class for featurizing torsions of selected atoms"""

    def __init__(self, ensemble_handler: EnsembleHandler, sel: str):
        super().__init__(ensemble_handler)
        self.feature_type = "Torsions"
        self.selection_string = sel

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr_path = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc_path = self.ensemble_handler.path_dict[ensemble]["xtc"]
            name, data = read_BAT(
                tpr_path,
                xtc_path,
                sel=self.selection_string,
            )
            ensemble_feature_df = pd.DataFrame(data, columns=name)
            # Keep only the torsions columns with BAT_t_ in the name
            ensemble_feature_df = ensemble_feature_df.filter(regex="BAT_t_", axis=1)
            # Replace BAT_t with TORS
            ensemble_feature_df.columns = [
                col.replace("BAT_t_", "TORS_") for col in ensemble_feature_df.columns
            ]
            # Add the timestep and ensemble columns
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class BPWaterFeaturizer(BaseFeaturizer):
    """Class for parameterizing the number of water molecules within a given radius of a com coordinate"""

    def __init__(self, ensemble_handler: EnsembleHandler):
        super().__init__(ensemble_handler)
        self.feature_type = "WaterOccupancy"

    def featurize(self, radius, n_jobs=20, pbc_corrections=False):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            u = self.ensemble_handler.get_universe_dict()[ensemble]

            if pbc_corrections:
                protein = u.select_atoms("protein")
                water = u.select_atoms("resname SOL")

                # Apply transformations
                workflow = [
                    trans.unwrap(u.atoms),
                    trans.center_in_box(protein, center="geometry"),
                    trans.wrap(water, compound="residues"),
                    trans.fit_rot_trans(protein, protein, weights="mass"),
                ]
                u.trajectory.add_transformations(*workflow)

            # Perform the analysis
            results = WaterOccupancyAnalysis(
                u.atoms,
                bp_selection_string=self.ensemble_handler.path_dict[ensemble][
                    "bp_selection_string"
                ],
                radius=radius,
                verbose=True,
            ).run()
            # Add to the df list
            ensemble_feature_df = results.df
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class WaterBridgeFeaturizer(BaseFeaturizer):
    def __init__(
        self, ensemble_handler: EnsembleHandler, sel1: str, sel2: str, order=1
    ):
        super().__init__(ensemble_handler)
        self.feature_type = "WaterBridge"
        self.selection_string1 = sel1
        self.selection_string2 = sel2
        self.order = order
        self.feature_dict = {}
        self.acceptors = []
        self.donors = []

    def get_hbond_donors_and_acceptors(self, ligand_selection: str):
        """
        This method should return the donors and acceptors for the water bridge analysis.
        """
        # Get the first non apo universe from the ensemble handler
        for ensemble in self.ensemble_handler.path_dict.keys():
            if "apo" not in ensemble:
                break
        u = self.ensemble_handler.get_universe_dict()[ensemble]

        # Get the ligand selection
        ligand = u.select_atoms(ligand_selection)

        # Standard valence electrons
        valence_electrons_dict = {
            "H": 1,
            "C": 4,
            "N": 5,
            "O": 6,
            "F": 7,
            "Cl": 7,
            "Br": 7,
            "I": 7,
            "S": 6,
            "P": 5,
        }

        # Typical valences for bonding
        expected_bonds_dict = {
            "H": 1,
            "C": 4,
            "N": 3,
            "O": 2,
            "F": 1,
            "Cl": 1,
            "Br": 1,
            "I": 1,
            "S": 2,  # varies
            "P": 3,  # varies
        }

        # Define electronegative elements
        electronegga = ["O", "N", "S", "F", "Cl", "Br", "P"]

        for atom in ligand.atoms:
            # Get element type from atom name
            elem = guess_atom_element(atom.name)
            # print(f"Atom: {atom.name}, Type: {atom.type}, Element: {elem}")
            if elem == "H":
                # Check if it is bonded to an electronegative atom
                bonded_atoms = atom.bonded_atoms
                for bonded_atom in bonded_atoms:
                    if guess_atom_element(bonded_atom.name) in electronegga:
                        # If it is bonded to an electronegative atom, it is a donor
                        self.donors.append(atom.name)
                        # print(f"Donor: {atom.name}")
                        break
            if elem in electronegga:
                # Check if it has lone pairs
                elem = guess_atom_element(atom.name)
                # Guessed element
                # print(f"Element guessed: {elem}")
                bonded = len(atom.bonds)
                valence = valence_electrons_dict.get(elem, 0)
                expected = expected_bonds_dict.get(elem, 0)

                lone_pair_electrons = valence - bonded
                lone_pairs = lone_pair_electrons // 2 if lone_pair_electrons >= 0 else 0

                if lone_pairs > 0:
                    self.acceptors.append(atom.name)
                    # print(f"Acceptor: {atom.name} with {bonded} bonds and  {lone_pairs} lone pairs")
        # Print the donors and acceptors
        print("Donors:", self.donors)
        print("Acceptors:", self.acceptors)

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            u = self.ensemble_handler.get_universe_dict()[ensemble]

            # Undo PBC
            water = u.select_atoms("resname SOL")
            protein = u.select_atoms("protein")
            workflow = [
                trans.unwrap(u.atoms),  # Unwrap all fragments
                trans.center_in_box(protein, center="geometry"),  # Center the protein
                trans.wrap(
                    water, compound="residues"
                ),  # Wrap water molecules back into the box
                trans.fit_rot_trans(
                    protein, protein, weights="mass"
                ),  # Align protein to the first frame
            ]
            u.trajectory.add_transformations(*workflow)

            # Perform the analysis
            print(
                "selection1:",
                self.selection_string1,
                u.select_atoms(self.selection_string1).resnames,
            )
            print(
                "selection2:",
                self.selection_string2,
                u.select_atoms(self.selection_string2).resnames,
            )
            w = MDAnalysis.analysis.hydrogenbonds.WaterBridgeAnalysis(
                u,
                selection1=self.selection_string1,
                selection2=self.selection_string2,
                water_selection="resname SOL",
                order=self.order,
                donors=self.donors,
                acceptors=self.acceptors,
            )
            w.run()

            # Store the results in a dictionary
            self.feature_dict[ensemble] = w.results.timeseries

    def convert(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, tuple):
            return tuple(self.convert(i) for i in obj)
        elif isinstance(obj, list):
            return [self.convert(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self.convert(v) for k, v in obj.items()}
        else:
            return obj

    def save_features(self, output_dir: str, overwrite: Optional[bool] = False):
        """Save features to a json file"""
        out_file = os.path.join(output_dir, f"{self.feature_type}_features.json")
        if not os.path.exists(output_dir):
            logging.info("Creating output directory %s", output_dir)
            os.makedirs(output_dir)
        elif os.path.exists(out_file):
            logging.info(
                "Output directory %s already exists. Overwrite set to %s",
                output_dir,
                overwrite,
            )
            if not overwrite:
                raise ValueError(f"Output directory {output_dir} already exists.")
        # Convert the feature dictionary to a JSON serializable format
        with open(
            os.path.join(output_dir, f"{self.feature_type}_features.csv"), "w"
        ) as f:
            json.dump(self.convert(self.feature_dict), f)

    def load_features(self, input_dir: str):
        """Load features from a csv file"""
        json_file = os.path.join(input_dir, f"{self.feature_type}_features.json")
        with open(json_file, "r") as f:
            self.feature_dict = json.load(f)
        logging.info("Loaded features from %s", json_file)

    def get_feature_df(self):
        raise NotImplementedError(
            "get_feature_df() is not implemented for WaterBridgeFeaturizer. Use get_feature_dict() instead."
        )

    def get_feature_dict(self):
        """Return the feature dictionary"""
        return self.feature_dict


class HbondFeaturizer(BaseFeaturizer):
    def __init__(self, ensemble_handler: EnsembleHandler, sel1: str, sel2: str):
        super().__init__(ensemble_handler)
        self.feature_type = "Hbonds"
        self.selection_string1 = sel1
        self.selection_string2 = sel2
        self.feature_df = None

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc = self.ensemble_handler.path_dict[ensemble]["xtc"]
            names, data = read_h_bonds(
                tpr,
                xtc,
                selection1=self.selection_string1,
                selection2=self.selection_string2,
            )

            # Convert the output dictionary to a DataFrame
            print(data)
            ensemble_feature_df = pd.DataFrame(data=data.flatten(), columns=names)
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class WaterBindingSiteFeaturizer(BaseFeaturizer):
    def __init__(self, ensemble_handler: EnsembleHandler, sel1: str, sel2: str):
        super().__init__(ensemble_handler)
        self.feature_type = "WaterBindingSite"
        self.selection_string1 = sel1
        self.selection_string2 = sel2
        self.feature_df = None

    def featurize(self, selection_string_list: list[str]):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc = self.ensemble_handler.path_dict[ensemble]["xtc"]
            names, data = read_h_bonds(
                tpr,
                xtc,
                selection1=self.selection_string1,
                selection2=self.selection_string2,
            )

            # Convert the output dictionary to a DataFrame
            ensemble_feature_df = pd.DataFrame(data=data[0][0], columns=names)
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class SideChainTorsionsFeaturizer(BaseFeaturizer):
    def __init__(self, ensemble_handler: EnsembleHandler):
        super().__init__(ensemble_handler)
        self.feature_type = "SideChainTorsions"
        self.feature_df = None

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr_path = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc_path = self.ensemble_handler.path_dict[ensemble]["xtc"]
            name, data = read_protein_sidechain_torsions(
                tpr_path,
                xtc_path,
            )
            print(name)
            print(data)
            ensemble_feature_df = pd.DataFrame(data, columns=name)
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)


class LigandRMSDFeaturizer(BaseFeaturizer):
    def __init__(self, ensemble_handler: EnsembleHandler):
        super().__init__(ensemble_handler)
        self.feature_type = "LigandRMSD"
        self.feature_df = None

    def featurize(
        self, reference: MDAnalysis.Universe, lig_selection: str = "resname unk"
    ):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)

            # Get the universe for the ensemble
            ensemble_u = self.ensemble_handler.get_universe_dict()[ensemble]

            # Align the whole trajectory to the reference
            aligner = align.AlignTraj(
                ensemble_u, reference, select="protein and name CA", in_memory=True
            )
            aligner.run()

            # Select ligand in both universes
            ref_ligand = reference.select_atoms(lig_selection)
            ligand = ensemble_u.select_atoms(lig_selection)

            # Calculate RMSD of ligand to reference over all frames
            rmsd_values = []
            for ts in ensemble_u.trajectory:
                rmsd = rms.rmsd(
                    ligand.positions,
                    ref_ligand.positions,
                    center=True,
                    superposition=True,
                )
                rmsd_values.append(rmsd)

            # Store
            ensemble_feature_df = pd.DataFrame(
                {"frame": range(len(rmsd_values)), "ligand_rmsd_to_ref": rmsd_values}
            )

            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
