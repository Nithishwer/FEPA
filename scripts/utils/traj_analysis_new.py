import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD, GRO, XTC
from MDAnalysis.coordinates.memory import MemoryReader
import warnings
from matplotlib import pyplot as plt
import os
from glob import glob
from MDAnalysis.lib import distances
from MDAnalysis.analysis import rms, align
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
from sklearn import preprocessing
import umap
from scipy.spatial.distance import jensenshannon
# suppress some MDAnalysis warnings about PSF files
warnings.filterwarnings('ignore')


class trajDimRed:
    def __init__(self, universe_dict,saved=False):
        """
        Initialize with a dictionary of BP annotated trajectoris as mda universes. 
        """
        self.universe, self.frame_tags = self.combine_universes(universe_dict,saved)
        self.metadata = pd.DataFrame(self.frame_tags)
        self.sda_matrix = self.analyze(self.universe)
        self.js_histogram = {}
        self.k2ij = self.make_k2ij_dict(self.universe) # Converts k in sda to i,j where i is the first atom, j is the second atom

    def combine_universes(self, universe_dict, saved=False):
        """
        Combine multiple MDAnalysis Universe objects into a single Universe object.

        Parameters:
        - universe_dict: Dictionary where keys are names/identifiers and values are MDAnalysis Universe objects

        Returns:
        - Combined MDAnalysis Universe object
        """
        # Check if the dictionary is empty
        if not universe_dict:
            print("The universe dictionary is empty.")
            return None

        # Make dir called protein_trajectories if it is not there already
        if not os.path.exists('protein_trajectories'):
            os.makedirs('protein_trajectories')

        if saved==False:
            # Iterate over the dictionary and save only the c-alphas trajectories
            for key, u in universe_dict.items():
                ag = u.select_atoms(f'protein and name CA and segid BP')
                print(f"Writing protein trajectory for {key}")
                ag.write(f"protein_trajectories/{key}_bp_protein.pdb", frames = u.trajectory[:1])
                ag.write(f'protein_trajectories/{key}_bp_protein_all.xtc', frames='all')

        # Load the PDB and XTC files
        pdb_files = [f'protein_trajectories/{key}_bp_protein.pdb' for key,u in universe_dict.items()]
        xtc_files = [f'protein_trajectories/{key}_bp_protein_all.xtc' for key,u in universe_dict.items()]
        
        if self.check_consistency(pdb_files):
            # Combine all XTC trajectories into a single Universe if they are consistent
            reference_pdb = pdb_files[0]  # Take the first PDB file as the reference structure
            u = mda.Universe(reference_pdb, *xtc_files)
            frame_tags = self.tag_frames_with_source_and_timestep(pdb_files, xtc_files)
            print("Combined universe loaded successfully!")
            return u, frame_tags
        else:
            print("Consistency check failed. Universe not combined.")
            return None
    
    def check_consistency(self,pdb_files):
        """
        Checks if atom names and residue names are consistent across multiple PDB files.

        Parameters:
        pdb_files : list
            List of paths to PDB files.

        Returns:
        bool
            True if all atom names, residue names, and residue IDs match across all PDB files.
        """
        # Load the reference structure (first PDB file)
        reference_u = mda.Universe(pdb_files[0])
        reference_u = reference_u.select_atoms('segid BP')

        # Extract atom and residue information from the reference universe
        reference_atom_names = reference_u.atoms.names
        reference_resnames = reference_u.residues.resnames
        reference_resids = reference_u.residues.resids

        # Iterate through all other PDB files and check consistency
        for pdb_file in pdb_files[1:]:  # Start from second PDB file
            u = mda.Universe(pdb_file)
            u = u.select_atoms('segid BP')

            atom_names = u.atoms.names
            resnames = u.residues.resnames
            resids = u.residues.resids

            # if not (atom_names == reference_atom_names).all():
            #     print(f"Atom names mismatch in {pdb_file}")
            #     return False
            if not (resnames == reference_resnames).all():
                print(f"Residue names mismatch in {pdb_file}")
                return False
            # if not (resids == reference_resids).all():
            #     print(f"Residue IDs mismatch in {pdb_file}")
            #     return False

        print("All files are consistent!")
        return True

    def tag_frames_with_source_and_timestep(self,pdb_files, xtc_files):
        """
        Loads and tags frames with information about the source file and original timestep.

        Parameters:
        pdb_files : list
            List of paths to PDB files (topology).
        xtc_files : list
            List of paths to XTC files (trajectories).

        Returns:
        combined_universe : MDAnalysis.Universe
            Combined universe with tagged frames.
        frame_tags : list
            List where each element is a dictionary of tags for the corresponding frame index.
        """
        # Initialize a list to store frame tags
        frame_tags = []

        # Initialize frame counter for the combined universe
        frame_counter = 0

        # Iterate over PDB and XTC files to load and tag
        for pdb_file, xtc_file in zip(pdb_files, xtc_files):
            # Load the universe for the current file pair
            u = mda.Universe(pdb_file, xtc_file)

            # Iterate over each frame in the current trajectory
            for ts in u.trajectory:
                # Append a dictionary with the source file and original timestep to the list
                frame_tags.append({
                    'source_file': xtc_file,
                    'timestep': ts.time,
                    'simulation': os.path.basename(xtc_file)[:-19]
                })
                # Increment the global frame counter
                frame_counter += 1

        return frame_tags

    def make_k2ij_dict(self,universe):
        """
        Makes a dictionary converting sda k to i,j
        """
        # Get the no of atoms in the pocket residue selection
        pocket_res = universe.select_atoms("segid BP")
        pocket_res_ca = pocket_res.select_atoms('protein and name CA')
        n = len(pocket_res_ca)

        # Initialize empty stuff
        k2ij = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                k += 1
                k2ij[k]=(i,j)

        return k2ij

    def analyze(self,universe):
        """
        Analyze the trajectories to compute self-distance arrays for specified residues.
        """
        # Make the dictionary
        sda_matrix = []
        for ts in universe.trajectory:
            pocket_res = universe.select_atoms("segid BP")
            pocket_res_ca = pocket_res.select_atoms('protein and name CA')
            sda = distances.self_distance_array(pocket_res_ca.positions, box=None)
            sda_matrix.append(np.array(sda))
        #print(self.sda_matrix)
        return np.array(sda_matrix)