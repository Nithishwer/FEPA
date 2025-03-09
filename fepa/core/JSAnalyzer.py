import MDAnalysis as mda
import warnings
from matplotlib import pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import dist
from scipy.spatial.distance import jensenshannon
from MDAnalysis.transformations import PositionAverager
warnings.filterwarnings('ignore')
from .DimReducer import PCAReducer
import logging
import MDAnalysis as mda
from glob import glob


class JSAnalyzer:
    """
    Analyzes MD trajectories using Jensen-Shannon divergence.

    This class takes a DimReducer object (which should contain reduced dimensionality
    data from PCA, t-SNE, or UMAP) and computes Jensen-Shannon divergence between
    distributions of reduced dimensions for different groups of trajectories,
    defined by metadata variables. It also provides functionalities to plot histograms,
    RMSF, and generate restraints based on RMSF.
    """
    def __init__(self, dimreducer, saved=False):
        """
        Initialize JSAnalyzer with a DimReducer object.

        Parameters:
        - dimreducer (DimReducer): An instance of the DimReducer class, containing
          reduced dimensionality data and trajectory information.
        - saved (bool, optional): Flag indicating if the DimReducer object contains saved data.
          Defaults to False.
        """
        logging.info("Initializing JSAnalyzer...")
        if not isinstance(dimreducer, PCAReducer): # Basic type check, can be extended for other DimReducers if needed
            logging.warning("DimReducer provided is not a PCAReducer. Ensure compatibility.")

        self.sda_matrix = np.array(dimreducer.sda_matrix) # Reduced dimension data matrix (samples x dimensions)
        self.universe = dimreducer.universe # MDAnalysis Universe object for the trajectory
        self.dimred_df_w_metadata = dimreducer.get_output() # DataFrame with reduced dimensions and metadata
        self.js_histogram = {} # Dictionary to store Jensen-Shannon divergence results and histograms
        self.k2ij = dimreducer.k2ij # Mapping from combined index k to original pair indices (i, j) in SDA matrix
        logging.info("JSAnalyzer initialized successfully.")

    def plot_distance_histogram_from_universe(self, variable, grp1, grp2, resn1, resn2, resid1, resid2, residoffset=0, save_path=None):
        '''
        Plots a histogram of distances between two residues in two groups of trajectories.
        '''
        logging.info(f"Plotting distance histogram for variable: {variable}, groups: {grp1} vs {grp2}")

        # Get frame iterators for each group's trajectory subset
        frame_iterator_1 = self.universe.trajectory[(self.dimred_df_w_metadata[variable]==grp1).values]
        frame_iterator_2 = self.universe.trajectory[(self.dimred_df_w_metadata[variable]==grp2).values]

        # Select atoms for the two residues        
        atom_grp_1 = self.universe.select_atoms(f'resname {resn1} and resid {resid1} and name CA')
        logging.info(f"Selected {atom_grp_1.n_atoms} atoms for residue {resn1} {resid1}")
        atom_grp_2 = self.universe.select_atoms(f'resname {resn2} and resid {resid2} and name CA')
        logging.info(f"Selected {atom_grp_2.n_atoms} atoms for residue {resn2} {resid2}")

        if atom_grp_1.n_atoms == 0:
            raise ValueError(f"Residue {resn1} {resid1} not found in the Universe.")
        if atom_grp_2.n_atoms == 0:
            raise ValueError(f"Residue {resn2} {resid2} not found in the Universe.")
        
        distances_1 = []
        distances_2 = []
        
        for ts in frame_iterator_1:
            distances_1.append(dist(atom_grp_1,atom_grp_2)[2][0])
        for ts in frame_iterator_2:
            distances_2.append(dist(atom_grp_1,atom_grp_2)[2][0])

        plt.figure(figsize=(10, 8))
        sns.histplot(distances_1, color='blue', label=f'{grp1}', stat='density', alpha=0.5)
        sns.histplot(distances_2, color='red', label=f'{grp2}', stat='density', alpha=0.5)
        plt.title(f'Distance between {resn1} {resid1} and {resn2} {resid2} in {variable} {grp1} vs {grp2}')
        plt.xlabel('Distance ($\AA$)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title='Groups')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Distance histogram saved to: {save_path}")

    def plot_distance_from_universe(self, variable, grp1, grp2, resn1, resn2, resid1, resid2, residoffset=0, save_path=None):
        '''
        Plots a histogram of distances between two residues in two groups of trajectories.
        '''
        logging.info(f"Plotting distance histogram for variable: {variable}, groups: {grp1} vs {grp2}")

        # Get frame iterators for each group's trajectory subset
        frame_iterator_1 = self.universe.trajectory[(self.dimred_df_w_metadata[variable]==grp1).values]
        frame_iterator_2 = self.universe.trajectory[(self.dimred_df_w_metadata[variable]==grp2).values]

        # Get timestep:
        ts_1 = self.dimred_df_w_metadata[(self.dimred_df_w_metadata[variable]==grp1).values]['timestep']
        ts_2 = self.dimred_df_w_metadata[(self.dimred_df_w_metadata[variable]==grp2).values]['timestep']

        # Select atoms for the two residues        
        atom_grp_1 = self.universe.select_atoms(f'resname {resn1} and resid {resid1} and name CA')
        logging.info(f"Selected {atom_grp_1.n_atoms} atoms for residue {resn1} {resid1}")
        atom_grp_2 = self.universe.select_atoms(f'resname {resn2} and resid {resid2} and name CA')
        logging.info(f"Selected {atom_grp_2.n_atoms} atoms for residue {resn2} {resid2}")

        if atom_grp_1.n_atoms == 0:
            raise ValueError(f"Residue {resn1} {resid1} not found in the Universe.")
        if atom_grp_2.n_atoms == 0:
            raise ValueError(f"Residue {resn2} {resid2} not found in the Universe.")
        
        distances_1 = []
        distances_2 = []
        
        for ts in frame_iterator_1:
            distances_1.append(dist(atom_grp_1,atom_grp_2)[2][0])
        for ts in frame_iterator_2:
            distances_2.append(dist(atom_grp_1,atom_grp_2)[2][0])


        # Plot line plot of distance vs time
        plt.figure(figsize=(10, 8))
        plt.plot(ts_1, distances_1, color='blue', label=f'{grp1}')
        plt.plot(ts_2, distances_2, color='red', label=f'{grp2}')
        plt.title(f'Distance between {resn1} {resid1} and {resn2} {resid2} in {variable} {grp1} vs {grp2}')
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('Distance ($\AA$)', fontsize=12)        
        plt.legend(title='Groups')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Distance vs t plot saved to: {save_path}")

    def compute_jensen_shanon(self, variable, grp1, grp2, num_bins=50):
        """
        Compute the Jensen-Shannon divergence between two groups based on a metadata variable.

        This method calculates the Jensen-Shannon divergence for each reduced dimension
        between two groups of trajectories, defined by specific values of a given
        metadata variable (e.g., 'condition', 'ligand'). It generates histograms
        for each dimension and stores the Jensen-Shannon entropies and histograms.

        Parameters:
        - variable (str): The name of the metadata variable to group trajectories by.
        - grp1 (str or comparable): The value of the metadata variable defining the first group.
        - grp2 (str or comparable): The value of the metadata variable defining the second group.
        - num_bins (int, optional): The number of bins to use for histogram generation. Defaults to 50.

        Returns:
        - int: 0 upon successful computation.
        - Stores results in self.js_histogram dictionary.
        """
        logging.info(f"Computing Jensen-Shannon divergence for variable: {variable}, groups: {grp1} vs {grp2}")

        if variable not in self.dimred_df_w_metadata.columns:
            logging.info(f"Possible variables are: {self.dimred_df_w_metadata.columns}")
            raise ValueError(f"Variable {variable} not found in the DataFrame.")

        if grp1 not in self.dimred_df_w_metadata[variable].unique():
            logging.info(f"Possible values for {variable} are: {self.dimred_df_w_metadata[variable].unique()}")
            raise ValueError(f"Group {grp1} not found in the DataFrame for variable {variable}.")

        if grp2 not in self.dimred_df_w_metadata[variable].unique():
            logging.info(f"Possible values for {variable} are: {self.dimred_df_w_metadata[variable].unique()}")
            raise ValueError(f"Group {grp2} not found in the DataFrame for variable {variable}.")

        logging.debug(f"Subsetting data for groups {grp1} and {grp2} based on variable {variable}.")
        index_1 = self.dimred_df_w_metadata[variable]==grp1 # Boolean index for group 1
        index_2 = self.dimred_df_w_metadata[variable]==grp2 # Boolean index for group 2

        # Subsetting reduced dimensionality data matrix for each group
        sda_matrix_grp1 = self.sda_matrix[index_1.values]
        sda_matrix_grp2 = self.sda_matrix[index_2.values]
        logging.debug(f"Data matrices subsetted. Group 1 shape: {sda_matrix_grp1.shape}, Group 2 shape: {sda_matrix_grp2.shape}")

        # Initialize lists to store JS entropies and histograms for each dimension
        num_columns = sda_matrix_grp1.shape[1] # Number of pairwise distances
        js_entropies = []
        histograms = []

        logging.debug(f"Iterating through {num_columns} pairwise distances to compute JS divergence.")
        for col in range(num_columns):
            logging.debug(f"Processing pairwise distance: {col} that is {self.k2ij[col]} that is {self.sda_index2resid_string([col])[0]}")
            # Extract data for the current dimension for both groups
            col_grp1 = sda_matrix_grp1[:, col]
            col_grp2 = sda_matrix_grp2[:, col]

            # Determine bin edges based on the combined range of both groups for consistent histograms
            min_val = min(col_grp1.min(), col_grp2.min())
            max_val = max(col_grp1.max(), col_grp2.max())
            bins = np.linspace(min_val, max_val, num_bins + 1) # Create bin edges

            # Compute histograms (frequency counts) and convert to probability distributions
            p_hist, _ = np.histogram(col_grp1, bins=bins, density=True) # Histogram for group 1
            q_hist, _ = np.histogram(col_grp2, bins=bins, density=True) # Histogram for group 2

            # Add a small value to avoid zero probabilities, which can cause issues with JS divergence
            p_hist += 1e-10
            q_hist += 1e-10

            # Normalize histograms to get probability distributions
            p = p_hist / p_hist.sum()
            q = q_hist / q_hist.sum()

            # Compute Jensen-Shannon divergence and square it to get JS entropy
            js_divergence = jensenshannon(p, q) ** 2
            js_entropies.append(js_divergence)
            histograms.append((col_grp1, col_grp2, bins)) # Store histogram data for plotting

        # Store the results in the js_histogram dictionary
        self.js_histogram[ variable + '_' + str(grp1) + '_' + str(grp2) ] = (np.array(js_entropies), histograms)
        logging.info(f"Jensen-Shannon divergence computation complete for variable: {variable}, groups: {grp1} vs {grp2}. Results stored in self.js_histogram.")
        return 0

    def sda_index2resid_string(self, top_indices, residoffset=0):
        '''
        Converts a list of SDA matrix indices to a list of residue strings.

        This function takes a list of indices that correspond to columns in the SDA matrix
        and converts them into human-readable residue pair strings, using residue names
        and residue IDs from the binding pocket atoms in the MDAnalysis Universe.

        Parameters:
        - top_indices (list or np.ndarray): List of integer indices corresponding to SDA matrix columns.
        - residoffset (int, optional): Offset to add to residue IDs for output. Defaults to 0.

        Returns:
        - list: List of strings, where each string represents a residue pair in the format
          "ResnameResid-ResnameResid" (e.g., "LYS123-GLU456").
        '''
        logging.debug("Converting SDA indices to residue strings.")
        # Get the binding pocket residues C-alpha atoms for index mapping
        pocket_res_ca = self.universe.select_atoms('segid BP and name CA')
        resid_list = []
        for index in top_indices:
            i,j = self.k2ij[index] # Get the pair of indices from the combined SDA index
            # Construct residue pair string using resname and resid with offset
            resid_str = (str(pocket_res_ca[i].resname) + str(pocket_res_ca[i].resid + residoffset) + '-' +
                         str(pocket_res_ca[j].resname) + str(pocket_res_ca[j].resid + residoffset))
            resid_list.append(resid_str)
        logging.debug(f"Converted indices to residue strings: {resid_list}")
        return resid_list

    def plot_top_histograms(self, variable, grp1, grp2, top_n=20, residoffset=0, save_path=None):
        """
        Plots histograms of top N dimensions with the highest Jensen-Shannon divergence.

        For the given variable and groups, this function plots overlaid histograms and KDE plots
        for the top N dimensions (SDA matrix columns) with the highest Jensen-Shannon divergence
        between the two groups. It visualizes the distributions and differences for the most
        discriminating dimensions.

        Parameters:
        - variable (str): Metadata variable used for grouping.
        - grp1 (str or comparable): Value for group 1.
        - grp2 (str or comparable): Value for group 2.
        - top_n (int, optional): Number of top dimensions to plot. Defaults to 20.
        - residoffset (int, optional): Offset for residue IDs in plot titles. Defaults to 0.
        - save_path (str, optional): Path to save the plot image. If None, plot is shown interactively. Defaults to None.
        """
        logging.info(f"Plotting top {top_n} histograms for variable: {variable}, groups: {grp1} vs {grp2}")
        key = variable + '_' + str(grp1) + '_' + str(grp2) # Construct key to retrieve JS results
        if key in self.js_histogram:
            js_entropies, histograms = self.js_histogram[key] # Retrieve pre-computed results
        else:
            error_msg = f"JS Entropy for {variable} {grp1} {grp2} not found; please run compute_jensen_shanon first."
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Ensure js_entropies is a numpy array for sorting
        js_entropies = np.array(js_entropies)
        # Get indices of the top N dimensions with highest JS entropy
        top_indices = np.argsort(js_entropies)[-top_n:][::-1]
        # Convert SDA indices to residue pair strings for plot titles
        resid_list = self.sda_index2resid_string(top_indices, residoffset)

        rows = 5 # Fixed rows for subplot layout
        cols = 4 # Fixed columns for subplot layout

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20)) # Create figure and subplots
        axes = axes.flatten() # Flatten axes array for easy iteration

        logging.debug(f"Iterating to plot histograms for top {top_n} dimensions.")
        for i, idx in enumerate(top_indices):
            logging.debug(f"Plotting histogram for dimension index: {idx}, SDA index: {top_indices[i]}")
            col_grp1, col_grp2, bins = histograms[idx] # Retrieve histogram data for current dimension

            # Plot histograms for group 1 and group 2
            sns.histplot(col_grp1, bins=bins, kde=False, color='blue', label='Group 1', stat='density', alpha=0.5, ax=axes[i])
            sns.histplot(col_grp2, bins=bins, kde=False, color='red', label='Group 2', stat='density', alpha=0.5, ax=axes[i])

            # Plot Kernel Density Estimation (KDE) for group 1 and group 2
            sns.kdeplot(col_grp1, color='blue', label='Group 1 KDE', ax=axes[i])
            sns.kdeplot(col_grp2, color='red', label='Group 2 KDE', ax=axes[i])

            # Set subplot title with residue pair, dimension index, JS entropy, and mean difference
            axes[i].set_title(f'{resid_list[i]}; Column {idx} - JS Entropy: {js_entropies[idx]:.4f}\n d: {abs(np.mean(col_grp1) - np.mean(col_grp2)):.2f}')
            axes[i].legend() # Show legend for each subplot

        # Hide any unused subplots if top_n is less than total subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout() # Adjust subplot parameters for tight layout
        if save_path:
            logging.info(f"Saving histogram plot to: {save_path}")
            plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        #plt.show() # Commented out to avoid interactive display if saving
        plt.close() # Close the plot to free memory
        logging.info("Histogram plotting complete.")

    def compute_rmsf(self, u, frame_iterator, ref=None, align_selection='name CA'):
        """
        Computes Root Mean Square Fluctuation (RMSF) for a subset of trajectory frames.

        This method calculates the RMSF for a given frame iterator (subset of trajectory)
        after aligning the trajectory to a reference structure. If no reference is provided,
        it aligns to the average structure of the provided frames. It first writes a spliced
        trajectory for efficiency before loading and processing it for RMSF calculation.

        Parameters:
        - u (MDAnalysis.Universe): The MDAnalysis Universe object for the full trajectory.
        - frame_iterator (MDAnalysis.Timeseries.TimeseriesIterator): Iterator for the frames to compute RMSF on.
        - ref (MDAnalysis.Universe, optional): Reference Universe for alignment. If None, average structure is used. Defaults to None.
        - align_selection (str, optional): Atom selection string for alignment. Defaults to 'segid BP and name CA' (Binding Pocket C-alpha atoms).

        Returns:
        - tuple: A tuple containing:
          - MDAnalysis.AtomGroup: The AtomGroup used for RMSF calculation.
          - np.ndarray: RMSF values for each atom in the AtomGroup.
        """
        logging.info("Computing RMSF for trajectory subset.")
        # Save spliced trajectory for atoms of interest to optimize alignment and RMSF calculation
        atomgroup = u.select_atoms(align_selection) # Select atoms for splicing and alignment
        logging.debug(f"Selected {len(atomgroup)} atoms for spliced trajectory.")
        atomgroup.write('spliced_traj.xtc', frames=frame_iterator) # Write spliced trajectory to .xtc file
        atomgroup.write('spliced_traj.gro') # Write coordinate file (.gro) for the spliced trajectory
        logging.info(f"Spliced trajectory written to spliced_traj.xtc and spliced_traj.gro")

        # Load the spliced trajectory as a new MDAnalysis Universe
        u_spliced = mda.Universe('spliced_traj.gro', 'spliced_traj.xtc')
        logging.debug("Spliced trajectory loaded into a new Universe object.")

        if ref is None: # Align to average structure if no reference is provided
            logging.info("No reference structure provided, aligning to average structure.")
            average = align.AverageStructure(u_spliced, u_spliced, ref_frame=0).run() # Compute average structure
            ref_universe = average.results.universe # Get Universe of the average structure
        else:
            logging.info(f"Using provided reference structure for alignment: {ref.trajectory.filename}")
            ref_universe = ref

        aligner = align.AlignTraj(u_spliced, ref_universe, in_memory=True).run() # Align trajectory to reference
        logging.debug("Trajectory alignment complete.")
        atom_grp = u_spliced.atoms # AtomGroup for RMSF calculation (all atoms in spliced universe)
        R = rms.RMSF(atom_grp).run() # Calculate RMSF
        rmsf_values = R.results.rmsf # Extract RMSF values

        logging.info("RMSF computation complete.")
        return atom_grp, rmsf_values

    def plot_rmsf_subset(self, variable, grp1, grp2, method='pca', residoffset=0, save_path='rmsf.png', ref=None):
        """
        Plots RMSF for two trajectory subsets defined by a metadata variable.

        This function computes and plots the RMSF for two groups of trajectories,
        defined by a metadata variable and its values. It overlays the RMSF plots for
        both groups and highlights binding pocket residues. The RMSF is calculated based
        on a spliced trajectory subset and alignment to a reference structure or average structure.

        Parameters:
        - variable (str): Metadata variable for grouping.
        - grp1 (str or comparable): Value for group 1.
        - grp2 (str or comparable): Value for group 2.
        - method (str, optional): Method name (e.g., 'pca', 'tsne') for plot labeling. Defaults to 'pca'.
        - residoffset (int, optional): Offset for residue IDs in plot labels. Defaults to 0.
        - save_path (str, optional): Path to save the RMSF plot image. Defaults to 'rmsf.png'.
        - ref (MDAnalysis.Universe, optional): Reference Universe for RMSF alignment. Defaults to None (average structure).
        """
        logging.info(f"Plotting RMSF for variable: {variable}, groups: {grp1} vs {grp2}")
        # Indexing frames for each group based on metadata variable
        index_1 = self.dimred_df_w_metadata[variable]==grp1 # Boolean index for group 1
        index_2 = self.dimred_df_w_metadata[variable]==grp2 # Boolean index for group 2

        # Get frame iterators for each group's trajectory subset
        frame_iterator_1 = self.universe.trajectory[index_1.values]
        frame_iterator_2 = self.universe.trajectory[index_2.values]
        logging.debug(f"Frame iterators created for both groups.")

        # Compute RMSF for each group
        logging.info(f"Computing RMSF for group 1 ({grp1}).")
        u1_c_alphas, u1_rmsf = self.compute_rmsf(u=self.universe, frame_iterator=frame_iterator_1, ref=ref)
        logging.info(f"Computing RMSF for group 2 ({grp2}).")
        u2_c_alphas, u2_rmsf = self.compute_rmsf(u=self.universe, frame_iterator=frame_iterator_2, ref=ref)
        logging.debug("RMSF computed for both groups.")

        # Get residue IDs with offset for plotting
        bp_offset_resids = [i + residoffset for i in self.universe.select_atoms('segid BP and name CA').residues.resids]
        u1_offset_resids = [i + residoffset for i in u1_c_alphas.resids]
        u2_offset_resids = [i + residoffset for i in u2_c_alphas.resids]

        # Plotting RMSF data
        plt.figure(figsize=(10, 6))
        plt.plot(u1_offset_resids, u1_rmsf, label=f'{method}:{variable}-{grp1}', color='blue') # Plot RMSF for group 1
        plt.plot(u2_offset_resids, u2_rmsf, label=f'{method}:{variable}-{grp2}', color='red') # Plot RMSF for group 2

        # Shade areas corresponding to binding pocket residues for visual context
        for resid in bp_offset_resids:
            plt.axvspan(resid+0.1, resid + 0.9, color='grey', alpha=0.3) # Shade region for each BP residue
        if ref is None:
            plt.title(f'Root Mean Square Fluctuation (RMSF) Aligned to Average')
        else:
            plt.title(f'Root Mean Square Fluctuation (RMSF) Aligned to reference {ref.trajectory.filename}')

        if hasattr(self, 'top_JS_resid_list') and self.top_JS_resid_list:
            for resid in self.top_JS_resid_list:
                resname_resid_1 = resid.split('-')[0]
                resname_resid_2 = resid.split('-')[1]
                resid_1 = int(resname_resid_1[3:])
                resid_2 = int(resname_resid_2[3:])
                plt.axvspan(float(resid_1)+0.1, float(resid_1) + 0.9, color='yellow', alpha=0.1) # Shade region for each BP residue
                plt.axvspan(float(resid_2)+0.1, float(resid_2) + 0.9, color='yellow', alpha=0.1) # Shade region for each BP residue
        else:
            logger.warning('No top JS residues found, cannot highlight them')

        plt.xlabel('Residue number', fontsize=12)
        plt.ylabel('RMSF ($\AA$)', fontsize=12)
        plt.legend(title='Groups')
        plt.grid(True, linestyle='--', alpha=0.7) # Add grid for better readability

        plt.xlim(min(u1_offset_resids), max(u1_offset_resids))  # Set x-axis limits to residue range
        plt.ylim(0, max(max(u1_rmsf), max(u2_rmsf)) * 1.1)  # Set y-axis limits with padding

        plt.tight_layout() # Adjust layout
        plt.savefig(save_path, dpi=300) # Save the plot
        plt.close() # Close the plot
        logging.info(f"RMSF plot saved to: {save_path}")

    def plot_top_histograms_w_restraints(self, variable, grp1, grp2, a=1, ref1=None, ref2=None, top_n=20, residoffset=0, save_path=None, write_restraints=False, restraints_file_suffix=None):
        """
        Plots top histograms with restraints visualized, and optionally generates restraint files.

        This function extends plot_top_histograms by adding visualization of restraints based on RMSF.
        It plots histograms for top dimensions, overlays restraints as shaded areas, and optionally writes
        GROMACS position restraint files based on the RMSF of each group.

        Parameters:
        - variable (str): Metadata variable for grouping.
        - grp1 (str or comparable): Value for group 1.
        - grp2 (str or comparable): Value for group 2.
        - a (float, optional): Scaling factor for RMSF to determine restraint width. Defaults to 1.
        - ref1 (MDAnalysis.Universe, optional): Reference for RMSF calculation for group 1. Defaults to None.
        - ref2 (MDAnalysis.Universe, optional): Reference for RMSF calculation for group 2. Defaults to None.
        - top_n (int, optional): Number of top dimensions to plot. Defaults to 20.
        - residoffset (int, optional): Residue ID offset. Defaults to 0.
        - save_path (str, optional): Path to save the plot image. Defaults to None.
        - write_restraints (bool, optional): Flag to write GROMACS restraint files. Defaults to False.
        - restraints_file_suffix (str, optional): Suffix for restraint file names. Defaults to None.
        """
        logging.info(f"Plotting top histograms with restraints for variable: {variable}, groups: {grp1} vs {grp2}")
        key = variable + '_' + str(grp1) + '_' + str(grp2) # Construct key to retrieve JS results
        if key in self.js_histogram:
            js_entropies, histograms = self.js_histogram[key] # Retrieve pre-computed JS results
        else:
            error_msg = f"JS Entropy for {variable} {grp1} {grp2} not found; please run compute_jensen_shanon first."
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Ensure js_entropies is a numpy array for sorting
        js_entropies = np.array(js_entropies)
        # Get indices of top N dimensions by JS entropy
        top_indices = np.argsort(js_entropies)[-top_n:][::-1]
        # Convert SDA indices to residue strings for plot titles
        resid_list = self.sda_index2resid_string(top_indices, residoffset)
        self.top_JS_resid_list = resid_list

        rows = 5 # Fixed rows for subplot layout
        cols = 4 # Fixed columns for subplot layout

        # Indexing frames for each group
        index_1 = self.dimred_df_w_metadata[variable]==grp1 # Boolean index for group 1
        index_2 = self.dimred_df_w_metadata[variable]==grp2 # Boolean index for group 2
        # Get frame iterators for each group's trajectory subset
        frame_iterator_1 = self.universe.trajectory[index_1.values]
        frame_iterator_2 = self.universe.trajectory[index_2.values]
        logging.debug(f"Frame iterators created for both groups.")
        # Compute RMSF for each group - needed for restraint calculation
        logging.info(f"Computing RMSF for group 1 ({grp1}) to determine restraints.")
        u1_c_alphas, u1_rmsf = self.compute_rmsf(u=self.universe, frame_iterator=frame_iterator_1, ref=ref1)
        logging.info(f"Computing RMSF for group 2 ({grp2}) to determine restraints.")
        u2_c_alphas, u2_rmsf = self.compute_rmsf(u=self.universe, frame_iterator=frame_iterator_2, ref=ref2)
        logging.debug("RMSF computed for both groups for restraints.")

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20)) # Create figure and subplots
        axes = axes.flatten() # Flatten axes array

        logging.debug(f"Iterating to plot histograms with restraints for top {top_n} dimensions.")
        for i, idx in enumerate(top_indices):
            logging.debug(f"Plotting histogram with restraints for dimension index: {idx}, SDA index: {top_indices[i]}")
            col_grp1, col_grp2, bins = histograms[idx] # Retrieve histogram data

            # Calculate restraint range based on RMSF for group 1
            restr_i_u1, restr_j_u1 = self.get_restraints_from_sda_index(u1_rmsf, u1_c_alphas.resids, idx, a)
            restr_u1 = max(restr_i_u1, restr_j_u1) # Use max RMSF*a as restraint width

            # Calculate restraint range based on RMSF for group 2
            restr_i_u2, restr_j_u2 = self.get_restraints_from_sda_index(u2_rmsf, u2_c_alphas.resids, idx, a)
            restr_u2 = max(restr_i_u2, restr_j_u2) # Use max RMSF*a as restraint width

            # Calculate mean of each group's distribution
            if ref1 is None:
                grp1_mean = np.mean(col_grp1) # Mean for group 1
            else:
                raise NotImplementedError('ref1 not implemented cannot plot restraints with respect to a reference structure')
                # grp1_mean = self.analyze(ref1)[0][idx] # Placeholder for reference-based mean

            if ref2 is None:
                grp2_mean = np.mean(col_grp2) # Mean for group 2
            else:
                raise NotImplementedError('ref2 not implemented cannot plot restraints with respect to a reference structure')
                # grp2_mean = self.analyze(ref2)[0][idx] # Placeholder for reference-based mean

            # Define restraint boundaries as shaded areas around the mean
            lower_bound_u1 = grp1_mean - restr_u1
            upper_bound_u1 = grp1_mean + restr_u1

            lower_bound_u2 = grp2_mean - restr_u2
            upper_bound_u2 = grp2_mean + restr_u2

            # Plot restraint shaded areas
            axes[i].axvspan(xmin=lower_bound_u1, xmax=upper_bound_u1, color='blue', alpha=0.25, label=f'RMSF*a')
            axes[i].axvspan(xmin=lower_bound_u2, xmax=upper_bound_u2, color='red', alpha=0.25, label=f'RMSF*a')

            # Plot histograms for group 1 and group 2
            sns.histplot(col_grp2, bins=bins, kde=False, color='red', label=f'{grp2}', stat='density', alpha=0.5, ax=axes[i])
            sns.histplot(col_grp1, bins=bins, kde=False, color='blue', label=f'{grp1}', stat='density', alpha=0.5, ax=axes[i])

            # Plot mean lines for each group
            axes[i].axvline(grp1_mean, color='grey', linestyle='--')  # Mark mean for group 1
            axes[i].axvline(grp2_mean, color='grey', linestyle='--')  # Mark mean for group 2

            # Plot KDE for group 1 and group 2
            sns.kdeplot(col_grp2, color='red', label='Group 2 KDE', ax=axes[i])
            sns.kdeplot(col_grp1, color='blue', label='Group 1 KDE', ax=axes[i])

            # Set subplot title with residue pair, dimension index, JS entropy, and mean difference
            axes[i].set_title(f'{resid_list[i]}; Column {idx} - JS Entropy: {js_entropies[idx]:.4f}\n d: {abs(np.mean(col_grp1) - np.mean(col_grp2)):.2f}')
            axes[i].legend() # Show legend

        # Determine restraint file suffix if not provided
        if restraints_file_suffix is None:
            restraints_file_suffix = f'{variable}.itp'

        # Write restraint files if write_restraints flag is True
        if write_restraints:
            logging.info(f"Writing GROMACS restraint files for group 1 ({grp1}) and group 2 ({grp2}).")
            self.generate_gmx_restraints_file(u1_rmsf, a, output_file=f'ca_res_{grp1}_{restraints_file_suffix}') # Generate restraints for group 1
            self.generate_gmx_restraints_file(u2_rmsf, a, output_file=f'ca_res_{grp2}_{restraints_file_suffix}') # Generate restraints for group 2
            logging.info(f"Restraint files written: ca_res_{grp1}_{restraints_file_suffix}, ca_res_{grp2}_{restraints_file_suffix}")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout() # Adjust layout
        if save_path:
            logging.info(f"Saving histogram plot with restraints to: {save_path}")
            plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        #plt.show() # Commented out for non-interactive mode
        plt.close() # Close the plot
        logging.info("Histogram plotting with restraints complete.")

    def get_restraints_from_sda_index(self, rmsf_list, rmsf_list_resid, sda_index, a):
        """
        Calculates restraint values for a given SDA index based on RMSF.

        This function determines the restraint width for a given SDA matrix column index
        by looking up the RMSF values of the corresponding residues in the binding pocket.
        It scales the RMSF by a factor 'a' to determine the restraint size.

        Parameters:
        - rmsf_list (np.ndarray): Array of RMSF values for binding pocket C-alpha atoms.
        - rmsf_list_resid (np.ndarray): Array of residue IDs corresponding to the RMSF list.
        - sda_index (int): Index of the SDA matrix column.
        - a (float): Scaling factor for RMSF to determine restraint width.

        Returns:
        - tuple: A tuple containing two float values:
          - Restraint value for the first residue in the SDA pair.
          - Restraint value for the second residue in the SDA pair.
        """
        logging.debug(f"Calculating restraints from SDA index: {sda_index}")
        # Select C-alpha atoms in the binding pocket
        pocket_res_ca = self.universe.atoms.select_atoms('segid BP and name CA')
        # Get pair of indices (i, j) corresponding to the SDA index
        i, j = self.k2ij[sda_index]
        # Get residue name and ID for both residues in the pair
        resname_i = pocket_res_ca[i].resname
        resid_i = pocket_res_ca[i].resid
        resname_j = pocket_res_ca[j].resname
        resid_j = pocket_res_ca[j].resid

        # Find indices in the RMSF list corresponding to the residues from SDA index
        rmsf_index_i = np.where(rmsf_list_resid == resid_i)[0][0]
        rmsf_index_j = np.where(rmsf_list_resid == resid_j)[0][0]

        # Calculate and return restraint values by scaling RMSF with factor 'a'
        restr_val_i = rmsf_list[rmsf_index_i] * a
        restr_val_j = rmsf_list[rmsf_index_j] * a
        logging.debug(f"Restraint values calculated for SDA index {sda_index}: Residue {resid_i}: {restr_val_i:.3f}, Residue {resid_j}: {restr_val_j:.3f}")
        return restr_val_i, restr_val_j

    def generate_gmx_restraints_file(self, rmsf_list, a=1, output_file='ca_restraints.itp'):
        """
        Generates a GROMACS position restraints (.itp) file based on RMSF values.

        This function creates a GROMACS format .itp file that specifies position restraints
        for C-alpha atoms. The restraint force constant is fixed, and the restraint position
        is derived from the RMSF values scaled by factor 'a'. The generated file can be
        used in GROMACS simulations to apply position restraints based on flexibility.

        Parameters:
        - rmsf_list (np.ndarray): Array of RMSF values for C-alpha atoms.
        - a (float, optional): Scaling factor for RMSF to calculate restraint distance. Defaults to 1.
        - output_file (str, optional): Path to save the generated GROMACS restraint file. Defaults to 'ca_restraints.itp'.
        """
        logging.info(f"Generating GROMACS restraints file: {output_file}")
        # Select all C-alpha atoms in the system
        ca_atoms = self.universe.select_atoms('name CA')

        # Check if the length of the RMSF list matches the number of CA atoms
        if len(ca_atoms) != len(rmsf_list):
            warning_msg = 'Length of rmsf_list and number of CA atoms do not match. Restraint file might be incomplete.'
            logging.warning(warning_msg)
            print(warning_msg) # Print warning to console as well

        # Open the output restraint file in write mode
        with open(output_file, "w") as f:
            # Write the header for GROMACS position restraints
            f.write(f"{'[ position_restraints ]\n; atoms     functype  g   r     k'}\n")

            logging.debug(f"Writing restraints for {len(rmsf_list)} CA atoms.")
            # Write restraint entry for each C-alpha atom
            for i in range(len(rmsf_list)):
                atom_index = ca_atoms[i].index + 1 # GROMACS atom indices are 1-based
                restraint_value = rmsf_list[i] * 0.1 * a # Scale RMSF and convert to nm (approximate, assuming RMSF in Angstrom)
                # Write formatted line to the restraint file: atom_index, functype=2 (position restraint), group=1, r=restraint_value, k=force constant (fixed at 1000)
                f.write(f'{atom_index:<5}  {2:<10}  {1:<5}  {restraint_value:<8.3f}  {1000:<5}\n')
        logging.info(f"GROMACS restraints file generated: {output_file}")