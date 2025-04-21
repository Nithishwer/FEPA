"""
This module contains classes for featurizing molecular dynamics trajectories of multiple ensembles
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import multiprocessing
import numpy as np
from multiprocessing import Pool
from functools import partial
import MDAnalysis.analysis.hbonds
from pensa.features import read_atom_self_distances, read_h_bonds, read_protein_sidechain_torsions
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
            name, data = read_atom_self_distances(
                tpr_path,
                xtc_path,
                selection=bp_selection_string,
                step=1,
                naming="plain",
                transformations=pp_trans,
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

    def featurize(self, radius, n_jobs = 20, pbc_corrections = False):

        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            u = self.ensemble_handler.get_universe_dict()[ensemble]

            if pbc_corrections:
                protein = u.select_atoms('protein')
                water = u.select_atoms('resname SOL')

                # Apply transformations 
                workflow = [
                    trans.unwrap(u.atoms),
                    trans.center_in_box(protein, center='geometry'),
                    trans.wrap(water, compound='residues'),
                    trans.fit_rot_trans(protein, protein, weights='mass'),
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

    def __init__(self, ensemble_handler: EnsembleHandler, sel1:str, sel2: str, order = 1):
        super().__init__(ensemble_handler)
        self.feature_type = "WaterBridge"
        self.selection_string1 = sel1
        self.selection_string2 = sel2
        self.order = order
        self.feature_df = None

    def featurize(self):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            u = self.ensemble_handler.get_universe_dict()[ensemble]
            
            # Undo PBC
            water = u.select_atoms('resname SOL')
            protein = u.select_atoms('protein')
            workflow = [
                trans.unwrap(u.atoms),  # Unwrap all fragments
                trans.center_in_box(protein, center='geometry'),  # Center the protein
                trans.wrap(water, compound='residues'),  # Wrap water molecules back into the box
                trans.fit_rot_trans(protein, protein, weights='mass'),  # Align protein to the first frame
            ]
            u.trajectory.add_transformations(*workflow)

            # Perform the analysis
            print('selection1:', self.selection_string1, u.select_atoms(self.selection_string1).resnames)
            print('selection2:', self.selection_string2, u.select_atoms(self.selection_string2).resnames)
            w = MDAnalysis.analysis.hydrogenbonds.WaterBridgeAnalysis(u,selection1=self.selection_string1,selection2=self.selection_string2, order=self.order)
            w.run()
            print(w.results.timeseries)

            list_of_lists = w.results.timeseries
            # Check if any list in list of lists has an element
            if all(len(lst) == 0 for lst in list_of_lists):
                print("All lists are empty.")
            else:
                print('WATER BRIDGES FOUND!!!')
                print(list_of_lists)
                print('Now need to plot it!!!!!')

            # Create a parallel function
            # run_per_frame = partial(get_water_occupancy_in_sphere_per_frame,
            #             atomgroup=u.atoms,
            #             bp_selection_string=bp_selection_string,
            #             radius=radius)
            # frame_values = np.arange(u.trajectory.n_frames)

            # print(w.results.timeseries)

            # # Running analysis in parallel
            # with Pool(n_jobs) as worker_pool:
            #     result = worker_pool.map(run_per_frame, frame_values)

            # # Convert the output dictionary to a DataFrame
            # ensemble_feature_df = pd.DataFrame(data=result, columns=['n_waters'])
            # ensemble_feature_df["timestep"] = (
            #     self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            # )
            # ensemble_feature_df["ensemble"] = ensemble
            # feature_dfs.append(ensemble_feature_df)

        # self.feature_df = pd.concat(feature_dfs, ignore_index=True)

class HbondFeaturizer(BaseFeaturizer):
    def __init__(self, ensemble_handler: EnsembleHandler, sel1:str, sel2: str):
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
    def __init__(self, ensemble_handler: EnsembleHandler, sel1:str, sel2: str):
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