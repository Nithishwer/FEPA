import MDAnalysis as mda
import logging
import pandas as pd
import numpy as np
import MDAnalysis as mda
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup, Results


def get_water_occupancy_in_sphere_per_frame(frame_index, atomgroup, bp_selection_string, radius):

    # index the trajectory to set it to the frame_index frame
    atomgroup.universe.trajectory[frame_index]

    # coordinates change for each frame
    bp_atomgroup = atomgroup.select_atoms(bp_selection_string)
    com = bp_atomgroup.center_of_mass()
    logging.info(f"Center of Mass (COM) of the binding site: {com}")

    waters_near_com = atomgroup.select_atoms(f"resname SOL and point {com[0]} {com[1]} {com[2]} {radius}")
    n_waters = len(waters_near_com.residues)

    return n_waters

class WaterOccupancyAnalysis(AnalysisBase):
    # _analysis_algorithm_is_parallelizable = True

    # @classmethod
    # def get_supported_backends(cls):
    #     return ('serial', 'multiprocessing', 'dask',)

    # def _get_aggregator(self):
    #   return ResultsGroup(lookup={'timeseries': ResultsGroup.ndarray_vstack})

    def __init__(self, atomgroup, bp_selection_string, radius, verbose, **kwargs):
        trajectory = atomgroup.universe.trajectory
        super(WaterOccupancyAnalysis, self).__init__(trajectory,verbose=verbose)
        self.atomgroup = atomgroup
        self.bp_selection_string = bp_selection_string
        self.radius = radius
 
    def _prepare(self):
        '''
        Create array of zeros as placeholder for results
        '''
        self.results = np.zeros((self.n_frames,3)) # Columns for frameindex, timestep and Water occupancy
        # self.protein = self.atomgroup.select_atoms('protein')
        # self.water = self.atomgroup.select_atoms('resname SOL')

        # # Apply transformations 
        # print('self.atomgroup.universe', self.atomgroup.universe)
        # print('self.atomgroup', type(self.atomgroup))
        # workflow = [
        #     trans.unwrap(self.atomgroup.atoms),
        #     trans.center_in_box(self.protein, center='geometry'),
        #     trans.wrap(self.water, compound='residues'),
        #     trans.fit_rot_trans(self.protein, self.protein, weights='mass'),
        # ]
        # self.atomgroup.trajectory.add_transformations(*workflow)
        # self.results = Results()

    def _single_frame(self):
        occupancy = get_water_occupancy_in_sphere_per_frame(
            frame_index = self._frame_index,
            atomgroup=self.atomgroup,
            bp_selection_string=self.bp_selection_string,
            radius=self.radius,
        )
        self.results[self._frame_index,0] = self._ts.frame
        self.results[self._frame_index,1] = self._trajectory.time
        self.results[self._frame_index,2] = occupancy

    def _conclude(self):
        """
        Finish up by calculating an average and transforming our
        results into a DataFrame.
        """
        # by now self.result is fully populated
        columns = ['Frame', 'Time (ps)', 'occupancy']
        self.df = pd.DataFrame(self.results, columns=columns)
    


# def get_water_occupancy_in_sphere(xtc_path, tpr_path, bp_selection_string, radius):
#     """
#     Function to calculate water occupancy in a protein binding site.

#     Parameters:
#     -----------
#     xtc_path : str
#         Path to the XTC trajectory file.
#     tpr_path : str
#         Path to the TPR topology file.
#     com_residue_list : list of str
#         List of str specifying the binding site residue ids.
#         Used to get com of the residues which will be the center of the sphere

#     Returns:
#     --------
#     output_dict : dict
#         Dictionary containing water occupancy statistics including cumulative sum, 
#         cumulative mean, and atom count mean and sum in the binding site region.
#     """
#     # Load the universe (PDB and XTC trajectory)
#     u = mda.Universe(tpr_path, xtc_path)

#     # Select the protein and water molecules
#     protein = u.select_atoms('protein')
#     water = u.select_atoms('resname SOL')

#     # Set up transformations for PBC wrapping and protein alignment
#     workflow = [
#         trans.unwrap(u.atoms),  # Unwrap all fragments
#         trans.center_in_box(protein, center='geometry'),  # Center the protein
#         trans.wrap(water, compound='residues'),  # Wrap water molecules back into the box
#         trans.fit_rot_trans(protein, protein, weights='mass'),  # Align protein to the first frame
#     ]
#     u.trajectory.add_transformations(*workflow)

#     # Define array to store
#     n_waters_array = []

#     # Process the first frame of the trajectory
#     for ts in u.trajectory:
#         # Construct selection string for the binding site residues

#         c_alpha_atoms = u.select_atoms(bp_selection_string)

#         # Compute the center of mass (COM) of the binding site residues
#         com = c_alpha_atoms.center_of_mass()
#         print("Center of Mass (COM) of the binding site:", com)

#         waters_near_com = u.select_atoms(f"resname SOL and point {com[0]} {com[1]} {com[2]} {radius}")

#         n_waters_array.append(len(waters_near_com.residues))

#     return ['n_waters'], n_waters_array
