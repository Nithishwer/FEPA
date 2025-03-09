import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD, GRO, XTC
from MDAnalysis.coordinates.memory import MemoryReader
import warnings
from matplotlib import pyplot as plt
import os
from glob import glob
from MDAnalysis.lib import distances
from MDAnalysis.analysis import rms, align
# first, import nglview
import nglview as nv
import seaborn as sns
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

import MDAnalysis as mda
from glob import glob

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

    def do_dimred(self, method='pca', n_components=2, n_neighbors=15,min_dist=0.1):
        """
        Perform dimensionality reduction (PCA, t-SNE, or UMAP) on the sda_matrix and store the result.

        Parameters:
        - method: 'pca', 'tsne', or 'umap' (default is 'pca')
        - n_components: Number of components to reduce to (default is 4)
        """
        try:
            if method == 'pca':
                reducer = PCA(n_components=n_components)
                columns = [f'principal component {i + 1}' for i in range(n_components)]
                attribute_name = 'pca_df'
                success_message = "PCA success"
            elif method == 'tsne':
                reducer = TSNE(n_components=n_components)
                columns = [f'tsne component {i + 1}' for i in range(n_components)]
                attribute_name = 'tsne_df'
                success_message = "t-SNE success"
            elif method == 'umap':
                reducer = umap.UMAP(n_components=n_components,n_neighbors=n_neighbors,min_dist=min_dist)
                columns = [f'umap component {i + 1}' for i in range(n_components)]
                attribute_name = 'umap_df'
                success_message = "UMAP success"
            else:
                print("Invalid method. Choose 'pca', 'tsne', or 'umap'.")
                return

            components = reducer.fit_transform(self.sda_matrix)
            components_df = pd.DataFrame(data=components, columns=columns)
            setattr(self, attribute_name, pd.concat([components_df, self.metadata], axis=1))
            print(success_message)
        except Exception as e:
            print(f"{method.upper()} failed: {e}")

    def plot_dimred_sims(self, method='pca', targets=None, highlights=None, save_path=None):
        """
        Plot the results of dimensionality reduction (PCA or t-SNE or UMAP).

        Parameters:
        - method: 'pca', 'tsne', or 'umap' (default is 'pca')
        - targets: Specific targets to plot (default is None, which plots all targets)
        - highlights: Specific targets to highlight (default is None)
        - save_path: Path to save the figure (default is None, which does not save)
        """
        try:
            if method == 'pca':
                if not hasattr(self, 'pca_df') or self.pca_df is None:
                    print("PCA data not available. Please run do_dimensionality_reduction with pca first.")
                    return
                df = self.pca_df
                x_col = 'principal component 1'
                y_col = 'principal component 2'
                title = '2 Component PCA'
            elif method == 'tsne':
                if not hasattr(self, 'tsne_df') or self.tsne_df is None:
                    print("t-SNE data not available. Please run do_dimensionality_reduction with tsne first.")
                    return
                df = self.tsne_df
                x_col = 'tsne component 1'
                y_col = 'tsne component 2'
                title = '2 Component t-SNE'
            elif method == 'umap':
                if not hasattr(self, 'umap_df') or self.umap_df is None:
                    print("UMAP data not available. Please run do_dimensionality_reduction with umap first.")
                    return
                df = self.umap_df
                x_col = 'umap component 1'
                y_col = 'umap component 2'
                title = '2 Component UMAP'
            else:
                print("Invalid method. Choose 'pca' or 'tsne' or 'umap'")
                return
            if targets is None:
                targets = np.unique(df['simulation'])
            if highlights is None:
                highlights = []
            #sns.set(style="white", context="notebook")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel(x_col, fontsize=16)
            ax.set_ylabel(y_col, fontsize=16)
            ax.set_title(title, fontsize=20)
            colors = sns.color_palette("Set2", len(targets))
            for target, color in zip(targets, colors):
                indicesToKeep = df['simulation'] == target
                alpha = 1.0 if target in highlights else 0.5
                marker = 's' if target in highlights else 'o'
                size = 100 if target in highlights else 20
                ax.scatter(df.loc[indicesToKeep, x_col],
                           df.loc[indicesToKeep, y_col],
                           c=[color],
                           s=size,
                           alpha=alpha,
                           marker=marker,
                           edgecolor='w',
                           label=target)
                if target in highlights:
                    for i in df.loc[indicesToKeep].index:
                        ax.text(df.loc[i, x_col] - 2,
                                df.loc[i, y_col] + 0.5,
                                target,
                                fontsize=9,
                                fontweight='bold',
                                ha='left',  # Align text to the left of the point
                                va='bottom')  # Align text to the bottom of the point

            legend = ax.legend(title="Simulation", fontsize=12, title_fontsize='13')
            legend.get_frame().set_edgecolor('w')
            ax.grid(False)
            ax.set_facecolor('white')

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure
            #plt.show()
            plt.close()
            print(f"{title} plot success")
        except Exception as e:
            print(f"{method.upper()} plot failed: {e}")

        

    def plot_dimred_time(self, target, method='pca', save_path=None):
        """
        Plot the results of dimensionality reduction (PCA, t-SNE, or UMAP) over time.

        Parameters:
        - target: The target simulation to plot.
        - method: 'pca', 'tsne', or 'umap' (default is 'pca')
        """
        try:
            if method == 'pca':
                if not hasattr(self, 'pca_df') or self.pca_df is None:
                    print("PCA data not available. Please run do_pca first.")
                    return
                df = self.pca_df
                x_col = 'principal component 1'
                y_col = 'principal component 2'
                title = '2 Component PCA Over Time'
            elif method == 'tsne':
                if not hasattr(self, 'tsne_df') or self.tsne_df is None:
                    print("t-SNE data not available. Please run do_tsne first.")
                    return
                df = self.tsne_df
                x_col = 'tsne component 1'
                y_col = 'tsne component 2'
                title = '2 Component t-SNE Over Time'
            elif method == 'umap':
                if not hasattr(self, 'umap_df') or self.umap_df is None:
                    print("UMAP data not available. Please run do_umap first.")
                    return
                df = self.umap_df
                x_col = 'umap component 1'
                y_col = 'umap component 2'
                title = '2 Component UMAP Over Time'
            else:
                print("Invalid method. Choose 'pca', 'tsne', or 'umap'.")
                return

            sns.set(style="white", context="notebook")

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)

            ax.set_xlabel(x_col, fontsize=16)
            ax.set_ylabel(y_col, fontsize=16)
            ax.set_title(title, fontsize=20)

            # Get unique time values and set up color palette
            target_times = df.loc[df['simulation'] == target, 'timestep']
            times = np.unique(target_times)
            num_times = len(times)
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=times.min(), vmax=times.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            # Iterate over time points and plot
            for time in times:
                indicesToKeep = (df['simulation'] == target) & (df['timestep'] == time)
                ax.scatter(df.loc[indicesToKeep, x_col],
                           df.loc[indicesToKeep, y_col],
                           c=[cmap(norm(time))], alpha=0.5,
                           label=f'Time {time}')

            # Add colorbar
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('Time', fontsize=12)

            ax.grid(False)
            ax.set_facecolor('white')

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure
            #plt.show()
            plt.close()
        except Exception as e:
            print(f"{method.upper()} plot failed: {e}")


    def elbow_plot(self, method='pca',save_path=None):
        """
        Plot the elbow plot for determining the optimal number of clusters using PCA, t-SNE, or UMAP.

        Parameters:
        - method: 'pca', 'tsne', or 'umap' (default is 'pca')
        """
        try:
            if method == 'pca':
                if not hasattr(self, 'pca_df') or self.pca_df is None:
                    print("PCA data not available. Please run do_pca first.")
                    return
                df = self.pca_df
                columns = ['principal component 1', 'principal component 2',
                           'principal component 3', 'principal component 4']
                title = 'Elbow Plot for PCA'
            elif method == 'tsne':
                if not hasattr(self, 'tsne_df') or self.tsne_df is None:
                    print("t-SNE data not available. Please run do_tsne first.")
                    return
                df = self.tsne_df
                columns = ['tsne component 1', 'tsne component 2']
                title = 'Elbow Plot for t-SNE'
            elif method == 'umap':
                if not hasattr(self, 'umap_df') or self.umap_df is None:
                    print("UMAP data not available. Please run do_umap first.")
                    return
                df = self.umap_df
                columns = ['umap component 1', 'umap component 2',
                           'umap component 3', 'umap component 4']
                title = 'Elbow Plot for UMAP'
            else:
                print("Invalid method. Choose 'pca', 'tsne', or 'umap'.")
                return

            # Select the columns for clustering
            X = df[columns]

            # Normalize the data
            X_normalized = preprocessing.scale(X)

            # Calculate the sum of squared distances for different values of k
            sse = []
            for k in range(1, 20):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_normalized)
                sse.append(kmeans.inertia_)

            # Plot the elbow plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, 20), sse, marker='o', linestyle='-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Sum of Squared Distances')
            plt.title(title)
            plt.xticks(range(1, 20))
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure
            #plt.show()
            plt.close()
        except Exception as e:
            print(f"{method.upper()} elbow plot failed: {e}")


    def cluster(self, best_k, method='pca'):
        """
        Perform clustering on the results of dimensionality reduction (PCA, t-SNE, or UMAP).

        Parameters:
        - best_k: The optimal number of clusters for KMeans
        - method: 'pca', 'tsne', or 'umap' (default is 'pca')
        """
        try:
            if method == 'pca':
                if not hasattr(self, 'pca_df') or self.pca_df is None:
                    print("PCA data not available. Please run do_pca first.")
                    return
                df = self.pca_df
                columns = ['principal component 1', 'principal component 2',
                           'principal component 3', 'principal component 4']
                success_message = "PCA clustering success"
            elif method == 'tsne':
                if not hasattr(self, 'tsne_df') or self.tsne_df is None:
                    print("t-SNE data not available. Please run do_tsne first.")
                    return
                df = self.tsne_df
                columns = ['tsne component 1', 'tsne component 2']
                success_message = "t-SNE clustering success"
            elif method == 'umap':
                if not hasattr(self, 'umap_df') or self.umap_df is None:
                    print("UMAP data not available. Please run do_umap first.")
                    return
                clusterable_embedding = umap.UMAP(n_neighbors=30,min_dist=0.0,n_components=2,random_state=42).fit_transform(self.sda_matrix)
                labels = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=500).fit_predict(clusterable_embedding)
                clustered = (labels >= 0)
                df = self.umap_df
                df['cluster'] = labels
                columns = ['umap component 1', 'umap component 2',
                           'umap component 3', 'umap component 4']
                success_message = "UMAP clustering success"
                print(success_message)
                return

            else:
                print("Invalid method. Choose 'pca', 'tsne', or 'umap'.")
                return

            # Select the columns for clustering
            X = df[columns]

            # Normalize the data
            X_normalized = preprocessing.scale(X)

            # Perform KMeans clustering with the best number of clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            kmeans.fit(X_normalized)

            # Add the cluster labels to the DataFrame
            df['cluster'] = kmeans.labels_

            print(success_message)
            return df
        except Exception as e:
            print(f"{method.upper()} clustering failed: {e}")

    def plot_dimred_cluster(self, clusters=None, method='pca', palette = "husl", alpha=0.1, s=5, save_path=None):
        """
        Plot the clusters with enhanced aesthetics for PCA, t-SNE, or UMAP.

        Parameters:
        - clusters: List of cluster labels to plot (default is None)
        - method: 'pca', 'tsne', or 'umap' (default is 'pca')
        """
        try:
            if method == 'pca':
                if not hasattr(self, 'pca_df') or self.pca_df is None:
                    print("PCA data not available. Please run do_pca first.")
                    return
                df = self.pca_df
                x_col = 'principal component 1'
                y_col = 'principal component 2'
                title = 'Clustering Results for PCA'
            elif method == 'tsne':
                if not hasattr(self, 'tsne_df') or self.tsne_df is None:
                    print("t-SNE data not available. Please run do_tsne first.")
                    return
                df = self.tsne_df
                x_col = 'tsne component 1'
                y_col = 'tsne component 2'
                title = 'Clustering Results for t-SNE'
            elif method == 'umap':
                if not hasattr(self, 'umap_df') or self.umap_df is None:
                    print("UMAP data not available. Please run do_umap first.")
                    return
                df = self.umap_df
                x_col = 'umap component 1'
                y_col = 'umap component 2'
                title = 'Clustering Results for UMAP'
            else:
                print("Invalid method. Choose 'pca', 'tsne', or 'umap'.")
                return

            if clusters is None:
                clusters = df['cluster'].unique()

            # Plot the clusters with enhanced aesthetics
            plt.figure(figsize=(10, 6))
            colors = sns.color_palette(palette,len(clusters))  # Using a qualitative color palette
            for i, cluster in enumerate(clusters):
                cluster_data = df[df['cluster'] == cluster]
                plt.scatter(cluster_data[x_col], cluster_data[y_col],
                            label=f'Cluster {cluster}', s=s, edgecolors='w', alpha=alpha, c=[colors[i]])
            plt.xlabel(x_col, fontsize=14)
            plt.ylabel(y_col, fontsize=14)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(False)  # Removing grid lines for cleaner appearance
            plt.tight_layout()  # Adjust layout to prevent overlapping labels
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure
            #plt.show()
            plt.close()
        except Exception as e:
            print(f"{method.upper()} clustering plot failed: {e}")


    def compute_jensen_shanon(self,variable,grp1,grp2,num_bins=50, method='pca'):
        """
        Compute the Jensen-Shannon divergence between two groups based on PCA, t-SNE, or UMAP.

        Parameters:
        - variable: The variable for which groups are compared.
        - grp1: Value of the variable for group 1.
        - grp2: Value of the variable for group 2.
        - num_bins: Number of bins for histogram (default is 50).
        - method: 'pca', 'tsne', or 'umap' (default is 'pca').
        """

        if method == 'pca':
            if not hasattr(self, 'pca_df') or self.pca_df is None:
                print("PCA data not available. Please run do_pca first.")
                return
            index_1 = self.pca_df[variable]==grp1
            index_2 = self.pca_df[variable]==grp2
        if method == 'tsne':
            if not hasattr(self, 'tsne_df') or self.tsne_df is None:
                print("tsne data not available. Please run do_tsne first.")
                return
            index_1 = self.tsne_df[variable]==grp1
            index_2 = self.tsne_df[variable]==grp2
        if method == 'umap':
            if not hasattr(self, 'umap_df') or self.umap_df is None:
                print("UMAP data not available. Please run do_umap first.")
                return
            index_1 = self.umap_df[variable]==grp1
            index_2 = self.umap_df[variable]==grp2
            
        # Subsetting groups
        sda_matrix_grp1 = self.sda_matrix[index_1]
        sda_matrix_grp2 = self.sda_matrix[index_2]
        num_columns = sda_matrix_grp1.shape[1]
        js_entropies = []
        histograms = []
        js_histogram = {}
        for col in range(num_columns):
            # Extract the columns
            col_grp1 = sda_matrix_grp1[:, col]
            col_grp2 = sda_matrix_grp2[:, col]
            # Determine the bin edges based on the combined range of both columns
            min_val = min(col_grp1.min(), col_grp2.min())
            max_val = max(col_grp1.max(), col_grp2.max())
            bins = np.linspace(min_val, max_val, num_bins + 1)
            # Compute histograms (frequency counts) and convert them to probability distributions
            p_hist, _ = np.histogram(col_grp1, bins=bins, density=True)
            q_hist, _ = np.histogram(col_grp2, bins=bins, density=True)
            # Add a small value to avoid zero probabilities
            p_hist += 1e-10
            q_hist += 1e-10
            # Normalize to get probability distributions
            p = p_hist / p_hist.sum()
            q = q_hist / q_hist.sum()
            # Compute Jensen-Shannon divergence
            js_divergence = jensenshannon(p, q) ** 2  # JS divergence squared to get the JS entropy
            js_entropies.append(js_divergence)
            histograms.append((col_grp1, col_grp2, bins))
        self.js_histogram[ variable + '_' + str(grp1) + '_' + str(grp2) ] = (np.array(js_entropies), histograms)
        js_histogram[ variable + '_' + str(grp1) + '_' + str(grp2) ] = (np.array(js_entropies), histograms)
        return 0

    def sda_index2resid_string(self, top_indices, residoffset=0):
        '''
        Converts a list of sda index to list of string with resid and resnames
        '''
        # Get the pocket res selection
        pocket_res_ca = self.universe.select_atoms('segid BP and name CA')
        resid_list = []
        for index in top_indices:
            i,j = self.k2ij[index]
            #print(pocket_res_ca[i],pocket_res_ca[j])
            resid_list.append(str(pocket_res_ca[i].resname)+str(pocket_res_ca[i].resid+residoffset)+'-'+
                  str(pocket_res_ca[j].resname)+str(pocket_res_ca[j].resid+residoffset))
        #print(resid_list)
        return resid_list        

    def plot_top_histograms(self, variable,grp1,grp2,top_n=20,residoffset=0,method = 'pca', save_path=None):
        key = variable + '_' + str(grp1) + '_' + str(grp2)
        if key in self.js_histogram:
            js_entropies, histograms = self.js_histogram[key]
        else:
            self.compute_jensen_shanon(variable, grp1, grp2,method=method)
            js_entropies, histograms = self.js_histogram[key]

        # Ensure js_entropies is a numpy array
        js_entropies = np.array(js_entropies)
        # Get indices of the top N JS entropies
        top_indices = np.argsort(js_entropies)[-top_n:][::-1]
        # Get real resnames
        resid_list = self.sda_index2resid_string(top_indices,residoffset)

        rows = 5
        cols = 4

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        axes = axes.flatten()

        for i, idx in enumerate(top_indices):
            col_grp1, col_grp2, bins = histograms[idx]

            # Plot histograms
            sns.histplot(col_grp1, bins=bins, kde=False, color='blue', label='Group 1', stat='density', alpha=0.5, ax=axes[i])
            sns.histplot(col_grp2, bins=bins, kde=False, color='red', label='Group 2', stat='density', alpha=0.5, ax=axes[i])

            # Plot KDE
            sns.kdeplot(col_grp1, color='blue', label='Group 1 KDE', ax=axes[i])
            sns.kdeplot(col_grp2, color='red', label='Group 2 KDE', ax=axes[i])

            axes[i].set_title(f'{resid_list[i]}; Column {idx} - JS Entropy: {js_entropies[idx]:.4f}\n d: {abs(np.mean(col_grp1) - np.mean(col_grp2)):.2f}')
            axes[i].legend()

        # Hide any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_path:
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        #plt.show()
        plt.close()

    def write_sliced_trajectory(self,variable,key,method='pca'):
        subset_dir = 'sliced_trajectories/'
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
        # Define writefile names
        xtc_filename = subset_dir + variable + '_' + str(key) + '_' + method + '.xtc'
        pdb_filename = subset_dir + variable + '_' + str(key) + '_' + method + '.pdb'

        if method == 'pca':
            ag = self.universe.select_atoms(f'protein')
            index = self.pca_df[variable]==key
            ag.write(pdb_filename, frames = self.universe.trajectory[:1])
            ag.write(xtc_filename, frames=self.universe.trajectory[index.values])
            print(f"Writing protein trajectory for {key} at {xtc_filename} and {pdb_filename}")
        
        if method == 'tsne':
            ag = self.universe.select_atoms(f'protein')
            index = self.tsne_df[variable]==key
            ag.write(pdb_filename, frames = self.universe.trajectory[:1])
            ag.write(xtc_filename, frames=self.universe.trajectory[index.values])
            print(f"Writing protein trajectory for {key} at {xtc_filename} and {pdb_filename}")

        if method == 'umap':
            ag = self.universe.select_atoms(f'protein')
            index = self.umap_df[variable]==key
            ag.write(pdb_filename, frames = self.universe.trajectory[:1])
            ag.write(xtc_filename, frames=self.universe.trajectory[index.values])
            print(f"Writing protein trajectory for {key} at {xtc_filename} and {pdb_filename}")

    def read_sliced_trajectory(self,variable,key,method='pca'):
        subset_dir = 'sliced_trajectories/'
        xtc_filename = subset_dir + variable + '_' + str(key) + '_' + method + '.xtc'
        pdb_filename = subset_dir + variable + '_' + str(key) + '_' + method + '.pdb'

        if not os.path.exists(xtc_filename) or not os.path.exists(pdb_filename):
            print("No sliced trajectories found, writing them now")
            self.write_sliced_trajectory(variable,key,method)
        # Make the universe
        u = mda.Universe(pdb_filename, xtc_filename)
        return u

    def compute_rmsf(self,u, ref=None, align_selection='segid BP and name CA'):
        if ref is None:
            average = align.AverageStructure(u, u, select=align_selection,ref_frame=0).run()
            ref = average.universe
        aligner = align.AlignTraj(u, ref,select=align_selection,in_memory=True).run()
        c_alphas = u.select_atoms('protein and name CA')
        # Print the resids of all C-alpha atoms
        #for ca in c_alphas:
        #    print(ca.resid,ca.resname)
        R = rms.RMSF(c_alphas).run()
        return c_alphas, R.results.rmsf

    def plot_rmsf_subset(self, variable, key1, key2, method='pca',residoffset=0,save_path='rmsf.png',ref = None):
            # Read files now
            u1 = self.read_sliced_trajectory(variable, key1, method)
            u2 = self.read_sliced_trajectory(variable, key2, method)

            # Select frames that correspond to True in the boolean mask
            # print(f"Number of frames in group 1: {len(u1.trajectory)}")
            u1_c_alphas, u1_rmsf = self.compute_rmsf(u1,ref=ref)
            u2_c_alphas, u2_rmsf = self.compute_rmsf(u2,ref=ref)

            #print('u1_rmsf:',u1_rmsf)
            #print('u2_rmsf:',u2_rmsf)

            bp_offset_resids = [i + residoffset for i in u1_c_alphas.select_atoms('segid BP').residues.resids]
            u1_offset_resids = [i + residoffset for i in u1_c_alphas.resids]
            u2_offset_resids = [i + residoffset for i in u2_c_alphas.resids]

            # Plot the data with enhancements
            plt.figure(figsize=(10, 6))
            plt.plot(u1_offset_resids, u1_rmsf, label=f'{method}:{variable}-{key1}', color='blue')
            plt.plot(u2_offset_resids, u2_rmsf, label=f'{method}:{variable}-{key2}', color='red')

            # Shade bp_resid areas in grey
            for resid in bp_offset_resids:
                plt.axvspan(resid+0.1, resid + 0.9, color='grey', alpha=0.3)
            if ref==None:
                plt.title(f'Root Mean Square Fluctuation (RMSF) Aligned to Average')
            else:                
                plt.title(f'Root Mean Square Fluctuation (RMSF) Aligned to reference {ref.trajectory.filename}')
            plt.xlabel('Residue number', fontsize=12)
            plt.ylabel('RMSF ($\AA$)', fontsize=12)

            plt.legend(title='Groups')
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.xlim(min(u1_offset_resids), max(u1_offset_resids))  # Optional: adjust axis limits
            plt.ylim(0, max(max(u1_rmsf), max(u2_rmsf)) * 1.1)  # Add some padding to the y-axis

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
    
    def plot_top_histograms_w_restraints(self, variable, grp1, grp2,a=1, ref1=None,ref2=None, top_n=20,residoffset=0,method = 'pca', save_path=None, write_restraints=False, restraints_file_suffix = None):
        key = variable + '_' + str(grp1) + '_' + str(grp2)
        if key in self.js_histogram:
            js_entropies, histograms = self.js_histogram[key]
        else:
            self.compute_jensen_shanon(variable, grp1, grp2,method=method)
            js_entropies, histograms = self.js_histogram[key]

        # Ensure js_entropies is a numpy array
        js_entropies = np.array(js_entropies)
        # Get indices of the top N JS entropies
        top_indices = np.argsort(js_entropies)[-top_n:][::-1]
        # Get real resnames
        resid_list = self.sda_index2resid_string(top_indices,residoffset)

        rows = 5
        cols = 4

        # Get rmsf of the first group
        u1 = self.read_sliced_trajectory(variable, grp1, method)
        u1_c_alphas, u1_rmsf = self.compute_rmsf(u1, ref=ref1)
        
        # Get rmsf of the second group
        u2 = self.read_sliced_trajectory(variable, grp2, method)
        u2_c_alphas, u2_rmsf = self.compute_rmsf(u2, ref=ref2)

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        axes = axes.flatten()

        for i, idx in enumerate(top_indices):
            col_grp1, col_grp2, bins = histograms[idx]

            # Get restraints u1
            restr_i_u1, restr_j_u1 = self.get_restraints_from_sda_index(u1_rmsf, idx, a)
            #print('restr_i:',restr_i_u1)
            #print('restr_j:',restr_j_u1)
            restr_u1 = max(restr_i_u1, restr_j_u1)

            # Get restraints u2
            restr_i_u2, restr_j_u2 = self.get_restraints_from_sda_index(u2_rmsf, idx, a)
            #print('restr_i:',restr_i_u2)
            #print('restr_j:',restr_j_u2)
            restr_u2 = max( restr_i_u2, restr_j_u2)

            # Get mean
            if ref1 is None:
                grp1_mean = np.mean(col_grp1)
            else:
                grp1_mean = self.analyze(ref1)[0][idx]
            
            if ref2 is None:
                grp2_mean = np.mean(col_grp2)
            else:
                grp2_mean = self.analyze(ref2)[0][idx]

            # Define the boundaries for the shaded area
            lower_bound_u1 = grp1_mean - restr_u1
            upper_bound_u1 = grp1_mean + restr_u1

            # Define the boundaries for the shaded area
            lower_bound_u2 = grp2_mean - restr_u2
            upper_bound_u2 = grp2_mean + restr_u2

            # Plot Restraints
            axes[i].axvspan(xmin=lower_bound_u1, xmax=upper_bound_u1, color='blue', alpha=0.25, label=f'RMSF*a')
            axes[i].axvspan(xmin=lower_bound_u2, xmax=upper_bound_u2, color='red', alpha=0.25, label=f'RMSF*a')

            # Plot histograms
            sns.histplot(col_grp2, bins=bins, kde=False, color='red', label=f'{grp2}', stat='density', alpha=0.5, ax=axes[i])
            sns.histplot(col_grp1, bins=bins, kde=False, color='blue', label='{grp1}', stat='density', alpha=0.5, ax=axes[i])

            # Plot mean/ref
            axes[i].axvline(grp1_mean, color='grey', linestyle='--')  # Optional: to mark the mean line
            axes[i].axvline(grp2_mean, color='grey', linestyle='--')  # Optional: to mark the mean line

            # Plot KDE
            sns.kdeplot(col_grp2, color='red', label='Group 2 KDE', ax=axes[i])
            sns.kdeplot(col_grp1, color='blue', label='Group 1 KDE', ax=axes[i])

            axes[i].set_title(f'{resid_list[i]}; Column {idx} - JS Entropy: {js_entropies[idx]:.4f}\n d: {abs(np.mean(col_grp1) - np.mean(col_grp2)):.2f}')
            axes[i].legend()
        
        if restraints_file_suffix is None:
            restraints_file_suffix = f'{variable}_{method}.itp'

        # Write restraints if necessary
        if write_restraints:
            self.generate_gmx_restraints_file(u1_rmsf, a, output_file=f'ca_res_{grp1}_{restraints_file_suffix}')
            self.generate_gmx_restraints_file(u2_rmsf, a, output_file=f'ca_res_{grp2}_{restraints_file_suffix}')

        # Hide any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_path:
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        #plt.show()
        plt.close()

    def get_restraints_from_sda_index(self, rmsf_list, sda_index, a):
        # Define binding pocket residues
        pocket_res_ca = self.universe.select_atoms('segid BP and name CA')
        #print('sda_index:',sda_index)
        # Get the residue no. in the binding pocket from sda index
        i,j = self.k2ij[sda_index]
        resname_i = pocket_res_ca[i].resname
        resid_i = pocket_res_ca[i].resid
        #print('resname_i:',resname_i)
        #print('resid_i:',resid_i)
        resname_j = pocket_res_ca[j].resname
        resid_j = pocket_res_ca[j].resid
        #print('resname_j:',resname_j)
        #print('resid_j:',resid_j)
        # Convert resid to the index in the rmsf_list by subtracting the offset
        rmsf_index_i = resid_i  - self.universe.residues[0].resid
        rmsf_index_j = resid_j  - self.universe.residues[0].resid
        #print('rmsf_index_i:',rmsf_index_i)


        return rmsf_list[rmsf_index_i]*a, rmsf_list[rmsf_index_j]*a

    def generate_gmx_restraints_file(self, rmsf_list, a=1, output_file='ca_restraints.itp'):
        # Select CA atoms
        ca_atoms = self.universe.select_atoms('name CA')
        
        # Check if the length of rmsf_list matches the number of CA atoms
        if len(ca_atoms) != len(rmsf_list):
            print('Length of rmsf_list and number of CA atoms do not match')
            return
        
        # Open the restraint file in write mode
        with open(output_file, "w") as f:
            # Write header
            f.write(f"{'[ position_restraints ]\n; atoms     functype  g   r     k'}\n")
            
            # Write each value with consistent spacing
            for i in range(len(rmsf_list)):
                atom_index = ca_atoms[i].index + 1
                restraint_value = rmsf_list[i] * 0.1 * a
                # Format the output with specific widths to ensure alignment
                f.write(f'{atom_index:<5}  {2:<10}  {1:<5}  {restraint_value:<8.3f}  {1000:<5}\n')


        #restr_file = open(output_file, 'w')
        #for i, res in enumerate(pocket_res_ca):
        #    restr = rmsf_list[i]*a
        #    restr_file.write(f'{res.resid} {restr}\n')
        #restr_file.close()
        #print(f"Restraints file written to {output_file}")