"""
This module contains classes for featurizing molecular dynamics trajectories of multiple ensembles
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from pensa.features import read_atom_self_distances
from fepa.core.ensemble_handler import EnsembleHandler


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
        if not os.path.exists(output_dir):
            logging.info("Creating output directory %s", output_dir)
            os.makedirs(output_dir)
        else:
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
            pdb_path = self.ensemble_handler.path_dict[ensemble]["pdb"]
            xtc_path = self.ensemble_handler.path_dict[ensemble]["xtc"]
            bp_selection_string = self.ensemble_handler.path_dict[ensemble][
                "bp_selection_string"
            ]
            name, data = read_atom_self_distances(
                pdb_path,
                xtc_path,
                selection=bp_selection_string,
                step=1,
                naming="plain",
            )
            ensemble_feature_df = pd.DataFrame(data, columns=name)
            ensemble_feature_df["timestep"] = (
                self.ensemble_handler.get_timestep_from_universe(key=ensemble)
            )
            ensemble_feature_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_feature_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
