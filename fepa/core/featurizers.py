"""
This module contains classes for featurizing molecular dynamics trajectories of multiple ensembles
"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal
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
from collections import defaultdict
from typing import Literal
from scipy.spatial import ConvexHull, Delaunay

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

    def featurize(self, radius, resname='SOL', n_jobs=20, pbc_corrections=False):
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
                resname=resname
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

            # Print the shape of data and names for debugging
            logging.info(f"Data shape: {data.shape}, Names length: {len(names)}")
            logging.info(f"Names: {names[:5]}")
            logging.info(f"Data (first 5 rows): {data[:5]}")

            # Convert the output dictionary to a DataFrame        
            ensemble_feature_df = pd.DataFrame(data=data.T, columns=names)
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

    def featurize(self, selection: str = "all"):
        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing %s...", ensemble)
            tpr_path = self.ensemble_handler.path_dict[ensemble]["tpr"]
            xtc_path = self.ensemble_handler.path_dict[ensemble]["xtc"]
            name, data = read_protein_sidechain_torsions(
                tpr_path,
                xtc_path,
                selection=selection,
            )
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


class BindingPocketVolumeFeaturizer(BaseFeaturizer):
    """Featurizer that computes binding pocket volume per frame for each ensemble.

    Volume can be computed using either a convex hull or an alpha-shape (3D) over
    the positions of atoms specified by an MDAnalysis selection.

    Output schema (concatenated over all ensembles):
        - frame:   integer frame index within the trajectory
        - time_ps: time in picoseconds from MDAnalysis (ts.time)
        - pocket_volume: volume in Å^3 for the selected pocket atoms at that frame
        - method:  'convex' or 'alpha'
        - alpha:   float (alpha-shape radius threshold), NaN for convex method
        - timestep: timestep (ps) inferred from EnsembleHandler.get_timestep_from_universe
        - ensemble: ensemble key (string)
    """

    def __init__(self, ensemble_handler: EnsembleHandler):
        super().__init__(ensemble_handler)
        self.feature_type = "BindingPocketVolume"
        self.feature_df = None

    # ------------------ geometry helpers ------------------ #
    @staticmethod
    def _convex_hull_volume(points: np.ndarray) -> float:
        if points.shape[0] < 4:
            return 0.0
        try:
            return float(ConvexHull(points).volume)
        except Exception:
            return 0.0

    @staticmethod
    def _tetra_circumradius(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> float:
        A = np.vstack([b - a, c - a, d - a]).T  # 3x3
        try:
            sq = np.array(
                [
                    np.dot(b - a, b - a),
                    np.dot(c - a, c - a),
                    np.dot(d - a, d - a),
                ]
            )
            x = np.linalg.inv(A).dot(sq)
            circumcenter = 0.5 * (A.dot(x))
            return float(np.linalg.norm(circumcenter))
        except np.linalg.LinAlgError:
            return np.inf

    @classmethod
    def _alpha_shape_triangles(cls, points: np.ndarray, alpha: float):
        if points.shape[0] < 4:
            return []
        delaunay = Delaunay(points)
        tets = delaunay.simplices
        keep = np.zeros(len(tets), dtype=bool)

        # mark tetrahedra whose circumsphere radius < alpha
        for i, tet in enumerate(tets):
            a, b, c, d = points[tet]
            R = cls._tetra_circumradius(a, b, c, d)
            if R < alpha:
                keep[i] = True

        # boundary faces occur exactly once among kept tets


        face_count = defaultdict(int)
        face_owner = {}

        def fkey(i, j, k):
            return tuple(sorted((i, j, k)))

        for idx, k in enumerate(keep):
            if not k:
                continue
            tet = tets[idx]
            faces = [
                (tet[0], tet[1], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[3]),
                (tet[1], tet[2], tet[3]),
            ]
            for f in faces:
                key = fkey(*f)
                face_count[key] += 1
                face_owner[key] = f

        return [face_owner[k] for k, v in face_count.items() if v == 1]

    @classmethod
    def _alpha_shape_volume(cls, points: np.ndarray, alpha: float) -> float:
        if points.shape[0] < 4:
            return 0.0
        tris = cls._alpha_shape_triangles(points, alpha)
        if not tris:
            return 0.0
        vol = 0.0
        for i, j, k in tris:
            v0, v1, v2 = points[i], points[j], points[k]
            vol += np.dot(v0, np.cross(v1, v2)) / 6.0
        return float(abs(vol))

    # ------------------ public API ------------------ #
    def featurize(
        self,
        selection: Optional[str] = None,
        method: Literal["convex", "alpha"] = "alpha",
        alpha: float = 3.0,
        use_pp_transforms: bool = False,
        pbc_corrections: bool = False,
    ):
        """Compute pocket volume over time for each ensemble.

        Args:
            selection: MDAnalysis selection string. If None, falls back to
                `path_dict[ensemble]["bp_selection_string"]`.
            method: 'convex' (ConvexHull) or 'alpha' (alpha-shape surface).
            alpha: Alpha-shape radius threshold (Å); used only if method=='alpha'.
            use_pp_transforms: If True, applies ensemble_handler.make_ensemble_pp_trans(ensemble)
                to trajectory prior to measurement (helps with continuity/alignment).
            pbc_corrections: If True, applies an explicit unwrap/center/wrap/fit pipeline
                similar to other featurizers.
        """
        assert method in ("convex", "alpha"), "method must be 'convex' or 'alpha'"

        feature_dfs = []
        for ensemble in self.ensemble_handler.path_dict.keys():
            logging.info("Featurizing pocket volume for %s...", ensemble)
            u = self.ensemble_handler.get_universe_dict()[ensemble]

            # Optional transformations
            workflows = []
            if use_pp_transforms:
                try:
                    workflows.extend(
                        self.ensemble_handler.make_ensemble_pp_trans(ensemble)
                    )
                except Exception:
                    logging.debug(
                        "make_ensemble_pp_trans not available or failed; skipping"
                    )
            if pbc_corrections:
                protein = u.select_atoms("protein")
                water = u.select_atoms("resname SOL")
                workflows.extend(
                    [
                        trans.unwrap(u.atoms),
                        trans.center_in_box(protein, center="geometry"),
                        trans.wrap(water, compound="residues"),
                        trans.fit_rot_trans(protein, protein, weights="mass"),
                    ]
                )
            if workflows:
                u.trajectory.add_transformations(*workflows)

            # Selection
            sel_str = selection
            if sel_str is None:
                sel_str = self.ensemble_handler.path_dict[ensemble].get(
                    "bp_selection_string", None
                )
            if not sel_str:
                raise ValueError(
                    f"No selection provided and no 'bp_selection_string' found for ensemble '{ensemble}'."
                )

            sel = u.select_atoms(sel_str)
            if len(sel) == 0:
                raise ValueError(
                    f"Selection '{sel_str}' returned zero atoms for ensemble '{ensemble}'."
                )

            # Iterate frames and compute volumes
            volumes = []
            times = []
            frames = []

            for ts in u.trajectory:
                pts = sel.positions.copy()
                if method == "convex":
                    vol = self._convex_hull_volume(pts)
                else:
                    vol = self._alpha_shape_volume(pts, alpha)
                volumes.append(vol)
                times.append(float(ts.time) if ts.time is not None else np.nan)
                frames.append(int(ts.frame))

            ensemble_df = pd.DataFrame(
                {
                    "frame": frames,
                    "time_ps": times,
                    "pocket_volume": volumes,
                    "method": method,
                    "alpha": (np.nan if method == "convex" else float(alpha)),
                }
            )
            ensemble_df["timestep"] = self.ensemble_handler.get_timestep_from_universe(
                key=ensemble
            )
            ensemble_df["ensemble"] = ensemble
            feature_dfs.append(ensemble_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
        return self.feature_df
