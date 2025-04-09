"""
This module contains utility functions for processing MDAnalysis objects.
"""

import warnings
import logging
import os
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from MDAnalysis.analysis import align, rms
from MDAnalysis import transformations
import MDAnalysis as mda
from fepa.utils.feature_utils import get_resid_pairs_from_sdf_names


def annotate_binding_pocket(gro: str, xtc: str, bp_string: str) -> mda.Universe:
    """
    Annotates binding pocket residues in an MDAnalysis Universe by assigning segment IDs.

    Parameters:
    -----------
    gro : str
        Path to the GRO file containing molecular structure.
    bp_string : str
        Atom selection string (MDAnalysis format) for binding pocket residues.
    xtc : str, optional
        Path to the XTC trajectory file. If not provided, only the GRO file is used.

    Returns:
    --------
    mda.Universe
        Universe object with segment attributes:
        - 'BP' for binding pocket residues
        - 'NON_BP' for all other residues

    Raises:
    -------
    FileNotFoundError:
        If the specified GRO or XTC file does not exist.
    ValueError:
        If the selection string returns an empty selection.
    """
    warnings.warn(
        "This function is deprecated and will be removed in future versions.",
        DeprecationWarning,
    )

    # Validate file existence
    if not os.path.exists(gro):
        raise FileNotFoundError(f"GRO file not found: {gro}")
    if xtc and not os.path.exists(xtc):
        raise FileNotFoundError(f"XTC file not found: {xtc}")

    # Load universe
    universe = mda.Universe(gro, xtc) if xtc else mda.Universe(gro)
    logging.info("Loaded universe from %s%s", gro, f" and {xtc}" if xtc else "")

    # Select binding pocket atoms
    bp_atoms = universe.select_atoms(bp_string)
    if not bp_atoms:
        raise ValueError(
            f"Selection string '{bp_string}' returned no atoms. Check your query."
        )

    # Assign segment IDs
    bp_atoms.residues.segments = universe.add_Segment(segid="BP")
    non_bp_atoms = universe.select_atoms("not segid BP")
    non_bp_atoms.residues.segments = universe.add_Segment(segid="NOBP")

    logging.info(
        "Annotated %d atoms as 'BP' and %d as 'NOBP'.", len(bp_atoms), len(non_bp_atoms)
    )

    return universe


def save_universes(
    universe_dict: Dict[str, mda.Universe],
    selection_string: str,
    save_location: str = "../../data/raw/",
) -> None:
    """
    Saves the binding pocket (BP) trajectories from a dictionary of MDAnalysis Universes.

    Parameters:
    -----------
    universe_dict : Dict[str, mda.Universe]
        Dictionary where keys are trajectory names and values are MDAnalysis Universe objects.
    save_location : str, optional
        Directory to save the processed binding pocket trajectory files ('../../data/raw/').

    Returns:
    --------
    None
    """
    if not universe_dict:
        logging.warning("The universe dictionary is empty. No files will be saved.")
        return

    save_path = Path(save_location)
    save_path.mkdir(parents=True, exist_ok=True)

    for key, universe in universe_dict.items():
        ag = universe.select_atoms(selection_string)
        bp_pdb_path = save_path / f"{key}_bp_protein.pdb"
        bp_xtc_path = save_path / f"{key}_bp_protein_all.xtc"
        logging.info("Saving %s trajectory for: %s", selection_string, key)
        ag.write(str(bp_pdb_path), frames=universe.trajectory[:1])
        ag.write(str(bp_xtc_path), frames="all")
        logging.info("Saved: %s, %s", bp_pdb_path, bp_xtc_path)


def check_bp_residue_consistency(universe_dict: Dict[str, mda.Universe]) -> bool:
    """
    Checks if the residue names in the annotated binding pocket are consistent.
    """
    # Load the reference structure (first PDB file)
    reference_key = list(universe_dict.keys())[0]
    reference_u = universe_dict[reference_key]
    reference_u = reference_u.select_atoms("segid BP and protein")

    # Extract atom and residue information from the reference universe
    reference_atom_names = reference_u.atoms.names
    reference_resnames = reference_u.residues.resnames
    reference_resids = reference_u.residues.resids

    for key, u in universe_dict.items():
        u = u.select_atoms("segid BP and protein")
        atom_names = u.atoms.names
        resnames = u.residues.resnames
        resids = u.residues.resids

        if not (atom_names == reference_atom_names).all():
            logging.error("Atom names mismatch in %s", key)
            return False
        if not (resnames == reference_resnames).all():
            logging.error("Residue names mismatch in %s", key)
            return False
        if not (resids == reference_resids).all():
            logging.error("Residue IDs mismatch in %s", key)
            return False

    logging.info("All files are consistent!")
    return True


def get_ca_rmsf(pdb_path: str, xtc_path: str):
    """
    Calculate the Root Mean Square Fluctuation (RMSF) of residues in a trajectory.

    Parameters:
    -----------
    pdb_path : str
        Path to the GRO file containing the reference structure.
    xtc_path : str
        Path to the XTC trajectory file.

    Returns:
    --------
    Dict[str, float]
        Dictionary where keys are residue names and values are RMSF values.
    """
    # Load the trajectory
    logging.info("Loading universe from %s and %s", pdb_path, xtc_path)
    u = mda.Universe(pdb_path, xtc_path)

    # Getting averge structure
    logging.info("Calculating average structure...")
    average = align.AverageStructure(
        u, u, select="protein and name CA", ref_frame=0
    ).run()
    ref = average.results.universe

    # Align trajectory to average structure
    logging.info("Aligning trajectory to average structure...")
    align.AlignTraj(
        u,
        ref,
        select="protein and name CA",
        filename="aligned_traj.dcd",
        in_memory=False,
    ).run()
    u = mda.Universe(pdb_path, "aligned_traj.dcd")

    # Calculate RMSF
    c_alphas = u.select_atoms("protein and name CA")
    R = rms.RMSF(c_alphas).run()

    # Create a RMSF dictionary with resid as keys
    rmsf_dict = {resid: rmsf for resid, rmsf in zip(c_alphas.resids, R.results.rmsf)}

    # selection and rmsf
    return rmsf_dict


def plot_ca_rmsfs(
    rmsf_dict_list: list[dict],
    labels: list[str],
    pocket_residues: list[int],
    save_path: str,
):
    """
    Plot Cα RMSF values for multiple trajectories.
    """
    plt.figure(figsize=(10, 5))
    # Shade bp_resid areas in grey
    for resid in pocket_residues:
        plt.axvspan(resid + 0.1, resid + 0.9, color="grey", alpha=0.3)
    for rmsf_dict, label in zip(rmsf_dict_list, labels):
        plt.plot(rmsf_dict.keys(), rmsf_dict.values(), label=label)
    plt.xlabel("Residue ID")
    plt.ylabel("RMSF (Å)")
    plt.legend()
    plt.title("Cα RMSF")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def generate_restraint_dicts(
    relative_entropy_dict, rmsf_dict_u1, rmsf_dict_u2, scaling_factor=1.0
):
    """
    Generate restraints based on RMSF values
    """
    ca_name_pairs = relative_entropy_dict["name"]
    resid_pairs = get_resid_pairs_from_sdf_names(ca_name_pairs)
    restr_u1_dict = {}  # dict[resid] = restr
    restr_u2_dict = {}  # dict[resid] = restr
    for resid_pair in resid_pairs:
        restr_1 = max(
            rmsf_dict_u1[resid_pair[0]] * scaling_factor,
            rmsf_dict_u1[resid_pair[1]] * scaling_factor,
        )
        restr_2 = max(
            rmsf_dict_u2[resid_pair[0]] * scaling_factor,
            rmsf_dict_u2[resid_pair[1]] * scaling_factor,
        )
        restr_u1_dict[resid_pair] = restr_1
        restr_u2_dict[resid_pair] = restr_2
    return restr_u1_dict, restr_u2_dict


def generate_gmx_restraints_file(u, rmsf_dict, a=1, output_file="ca_restraints.itp"):
    """
    Generate position restraints file for GROMACS based on RMSF values
    """
    # Select CA atoms
    ca_atoms = u.select_atoms("name CA")

    # Check if the length of rmsf_list matches the number of CA atoms
    if len(ca_atoms) != len(rmsf_dict.values()):
        print("Length of rmsf_list and number of CA atoms do not match")
        return

    # Open the restraint file in write mode
    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("{'[ position_restraints ]\n; atoms     functype  g   r     k'}\n")

        # Write each value with consistent spacing
        for ca_atom in ca_atoms.atoms:
            atom_index = ca_atom.index + 1
            restraint_value = rmsf_dict[ca_atom.resid] * 0.1 * a
            # Format the output with specific widths to ensure alignment
            f.write(
                f"{atom_index:<5}  {2:<10}  {1:<5}  {restraint_value:<8.3f}  {1000:<5}\n"
            )


def write_traj_without_PBC_jumps(
    gro_path: str,
    xtc_path: str,
    output_gro_path: str = "output.gro",
    output_xtc_path: str = "output.xtc",
    centering_selection_string: str = "protein",
    saving_selection_string: str = "all",
):
    """
    Remove periodic boundary conditions from a trajectory.
    Ideally, dont rewrite the original trajectory, but create a new one.
    I rewrite because I am recklesssss
    """
    u = mda.Universe(gro_path, xtc_path)
    centering_selection = u.select_atoms(centering_selection_string)
    ag = u.select_atoms(saving_selection_string)
    # we will use mass as weights for the center calculation
    workflow = (
        transformations.unwrap(ag),
        transformations.center_in_box(centering_selection, center="mass"),
        transformations.wrap(ag, compound="fragments"),
    )
    u.trajectory.add_transformations(*workflow)
    ag.write(output_gro_path)
    ag.write(output_xtc_path, frames="all")


#     # print(f'{pca1.shape[0]}')
#     combine_args = ",".join([f"d{i}" for i in range(1, len(pca1) + 1)])
#     coefficients = ",".join(map(str, pca1))
#     dot_product_text = f"""  # Create the dot product


# dot: COMBINE ARG={combine_args} COEFFICIENTS={coefficients} PERIODIC=NO
# CV: MATHEVAL ARG=dot FUNC=10*x PERIODIC=NO"""
#     # print(dot_product_text)
#     plumed_text += dot_product_text + "\n"
#     # Save the output in colvar
#     save_text = "PRINT ARG=CV FILE=COLVAR STRIDE=1"
#     # print(save_text)
#     plumed_text += save_text + "\n"
#     # Restraints
#     restraint_text = f"""# Put position of restraints here for each window
# restraint: RESTRAINT ARG=CV AT=@replicas:$RESTRAINT_ARRAY KAPPA={kappa}
# PRINT ARG=restraint.* FILE=restr
# """
#     if "apo" in grp1 and "apo" in grp2:
#         print("Both groups have apo in it :(")
#         return 0
#     if "apo" not in grp1 and "apo" not in grp2:
#         print("Both groups dont have apo in it :(")
#         return 0
#     else:
#         if "apo" in grp1:
#             apo = grp1
#             holo = grp2
#         else:
#             apo = grp2
#             holo = grp1

#     # Check if apo has a lower mean than holo:
#     CV_holo = self.y_projection[holo]
#     CV_apo = self.y_projection[apo]
#     # If holo distribution has lower mean:
#     if np.mean(CV_holo) < np.mean(CV_apo):
#         print(
#             f"Group {holo} ({np.mean(CV_holo)} has a lower mean than group {apo} ({np.mean(CV_apo)}"
#         )
#         min_val = np.min(np.append(CV_holo, CV_apo))
#         max_val = np.max(np.append(CV_holo, CV_apo))
#         # Restraints should go from min to max; Simulations go holo to apo
#         restraint_centers = np.linspace(min_val, max_val, 24)
#     # Check if CV_a distribution is to the right of CV_b distribution
#     if np.mean(CV_apo) < np.mean(CV_holo):
#         print(
#             f"Group {apo} ({np.mean(CV_apo)}) has a lower mean than group {holo} ({np.mean(CV_holo)})"
#         )
#         min_val = np.min(np.append(CV_holo, CV_apo))
#         max_val = np.max(np.append(CV_holo, CV_apo))
#         # Restraints should go from max to min; Simulations go holo to apo
#         restraint_centers = np.linspace(min_val, max_val, 24)
#         restraint_centers = np.flip(restraint_centers)
#     restraint_centers_str = ",".join(map(str, restraint_centers))
#     # replace $RESTRAINT_ARRAY with the restraint_centers_str
#     restraint_text = re.sub(r"\$RESTRAINT_ARRAY", restraint_centers_str, restraint_text)
#     # Add the restraints to the plumed file
#     plumed_text += restraint_text
#     # Save the plumed file
#     if save_path:
#         with open(save_path, "w") as f:
#             f.write(plumed_text)
#         print(f"Plumed file saved at {save_path}")
