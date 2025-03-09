import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

def find_protein_residues_near_unk(gro_file, cutoff=6.0):
    # Load the structure
    u = mda.Universe(gro_file)
    
    # Select the UNK molecule
    unk = u.select_atoms("resname unk")
    
    if len(unk) == 0:
        print("No UNK molecule found in the structure.")
        return
    
    # Select protein atoms
    protein = u.select_atoms("protein")
    
    # Compute distance array between protein and UNK
    distances = distance_array(protein.positions, unk.positions)
    
    # Find protein atoms within cutoff distance
    close_atoms = protein[distances.min(axis=1) < cutoff]
    
    # Get unique residue information
    unique_residues = {(res.resid, res.resname) for res in close_atoms.residues}

    # Create resno list
    resno_list = []
    
    # Print the results
    print("Residues within 6 Å of unk:")
    for resid, resname in sorted(unique_residues):
        print(f"Residue ID: {resid}, Residue Name: {resname}")
        resno_list.append(resid)

    print(f"Residue numbers: {resno_list}")

# Example usage
gro_file = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/OX2_42922/vanilla/prod.gro"  # Replace with your actual file path
find_protein_residues_near_unk(gro_file)