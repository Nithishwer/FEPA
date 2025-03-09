import os
import logging
from typing import Dict
import MDAnalysis as mda
from pathlib import Path
import warnings


def annotate_binding_pocket(gro: str,  xtc: str, bp_string: str) -> mda.Universe:
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
    warnings.warn("This function is deprecated and will be removed in future versions.", DeprecationWarning)
    
    # Validate file existence
    if not os.path.exists(gro):
        raise FileNotFoundError(f"GRO file not found: {gro}")
    if xtc and not os.path.exists(xtc):
        raise FileNotFoundError(f"XTC file not found: {xtc}")

    # Load universe
    universe = mda.Universe(gro, xtc) if xtc else mda.Universe(gro)
    logging.info(f"Loaded universe from {gro}" + (f" and {xtc}" if xtc else ""))

    # Select binding pocket atoms
    bp_atoms = universe.select_atoms(bp_string)
    if not bp_atoms:
        raise ValueError(f"Selection string '{bp_string}' returned no atoms. Check your query.")

    # Assign segment IDs
    bp_atoms.residues.segments = universe.add_Segment(segid="BP")
    non_bp_atoms = universe.select_atoms("not segid BP")
    non_bp_atoms.residues.segments = universe.add_Segment(segid="NOBP")
    
    logging.info(f"Annotated {len(bp_atoms)} atoms as 'BP' and {len(non_bp_atoms)} as 'NOBP'.")

    return universe


def save_universes(universe_dict: Dict[str, mda.Universe], selection_string: str, save_location: str = "../../data/raw/") -> None:
    """
    Saves the binding pocket (BP) trajectories from a dictionary of MDAnalysis Universes.

    Parameters:
    -----------
    universe_dict : Dict[str, mda.Universe]
        Dictionary where keys are trajectory names and values are MDAnalysis Universe objects.
    save_location : str, optional
        Directory to save the processed binding pocket trajectory files (default: '../../data/raw/').

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
        try:
            ag = universe.select_atoms(selection_string)

            bp_pdb_path = save_path / f"{key}_bp_protein.pdb"
            bp_xtc_path = save_path / f"{key}_bp_protein_all.xtc"

            logging.info(f"Saving {selection_string} trajectory for: {key}")

            ag.write(str(bp_pdb_path), frames=universe.trajectory[:1])
            ag.write(str(bp_xtc_path), frames="all")

            logging.info(f"Saved: {bp_pdb_path}, {bp_xtc_path}")

        except Exception as e:
            logging.error(f"Error processing {key}: {e}")

