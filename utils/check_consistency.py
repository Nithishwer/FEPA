import logging
import MDAnalysis as mda

def check_consistency(pdb_files, selection_string=None):
    """
    Checks if atom names and residue names are consistent across multiple PDB files.
    
    Parameters:
    pdb_files : list
        List of paths to PDB files.
    
    Returns:
    bool
        True if all atom names, residue names, and residue IDs match across all PDB files.
    """
    if not pdb_files:
        logging.error("No PDB files provided.")
        return False

    try:
        # Load the reference structure (first PDB file)
        reference_u = mda.Universe(pdb_files[0])
        if selection_string:
            reference_u = reference_u.select_atoms(selection_string)
        
        # Extract atom and residue information from the reference universe
        reference_atom_names = reference_u.atoms.names
        reference_resnames = reference_u.residues.resnames
        reference_resids = reference_u.residues.resids
        
        # Iterate through all other PDB files and check consistency
        for pdb_file in pdb_files[1:]:  # Start from second PDB file
            u = mda.Universe(pdb_file)
            atom_names = u.atoms.names
            resnames = u.residues.resnames
            resids = u.residues.resids
            
            if not (atom_names == reference_atom_names).all():
                logging.error(f"Atom names mismatch in {pdb_file}")
                return False
            if not (resnames == reference_resnames).all():
                logging.error(f"Residue names mismatch in {pdb_file}")
                return False
            if not (resids == reference_resids).all():
                logging.error(f"Residue IDs mismatch in {pdb_file}")
                return False
        
        logging.info("All files are consistent!")
        return True
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False