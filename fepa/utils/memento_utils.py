import os
import shutil
import logging
import MDAnalysis as mda
from PyMEMENTO import MEMENTO


def set_cys(in_resn, out_resn, indices, input_pdb, output_pdb):
    """
    Replace residue name in_resn with out_resn at specified residue indices in a PDB file.

    Parameters:
        in_resn (str): The residue name to be replaced.
        out_resn (str): The new residue name.
        indices (list of int): List of residue indices to apply the replacement.
        input_pdb (str): Path to the input PDB file.
        output_pdb (str): Path where the modified PDB file will be saved.
    """
    with open(input_pdb, "r") as infile, open(output_pdb, "w") as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                resn = line[17:20].strip()
                resi = int(line[22:26])
                if resn == in_resn and resi in indices:
                    # Replace the residue name (column 18-20 in PDB, 0-based 17-20)
                    newline = line[:17] + f"{out_resn:>3}" + line[20:]
                    outfile.write(newline)
                else:
                    outfile.write(line)
            else:
                outfile.write(line)


def run_pymemento(
    last_run=None,
    template_path=None,
    protonation_states=None,
    n_residues=None,
    cyx_indices=None,
):
    if template_path is None:
        logging.error("Template path is not defined.")
        return
    if n_residues is None:
        logging.error("Number of residues is not defined.")
        return
    # Define base path and file paths using the input string
    initial_gro = "initial.gro"  # -> no lipids
    target_gro = "target.gro"  # -> has membrane
    forcefield_paths = [f"{template_path}/amber99sb-star-ildn-mut.ff"]

    # Set up MEMENTO class
    model = MEMENTO(
        "wdir/",
        initial_gro,
        target_gro,
        list(
            range(1, n_residues + 1)
        ),  # Should go from 1 to n+1 where n is the number of residues in the protein
        forcefield="Other",
        lipid="resname PA or resname PC or resname OL",
        forcefield_paths=forcefield_paths,
        last_step_performed=last_run,
    )

    # Perform morphing and modeling
    model.morph(24)
    model.make_models(5)

    # Find best path and process models
    model.find_best_path(poolsize=1)  # Avoid multiprocessing due to memory bug

    if cyx_indices != None:
        # Loop through best pdbs in modeller
        for i in range(24):
            # Best pdb path
            best_path = os.path.join("wdir", "modeller", f"morph{i}", "best.pdb")
            # call a function to set the cys in the given indices to cyx
            shutil.copy(best_path, best_path.replace(".pdb", "_og.pdb"))
            set_cys(
                in_resn="CYS",
                out_resn="CYX",
                indices=cyx_indices,
                input_pdb=best_path.replace(".pdb", "_og.pdb"),
                output_pdb=best_path,
            )

    model.process_models(
        caps=False,
        his=True,
        his_protonation_states=protonation_states,
    )

    # Prepare and solvate boxes, then minimize
    model.prepare_boxes(template_path)
    model.solvate_boxes(ion_concentration=0.15)
    model.minimize_boxes()


def prepare_input_structures(initial_gro, target_gro):
    # Load both structures using MDAnalysis
    initial_universe = mda.Universe(initial_gro)
    target_universe = mda.Universe(target_gro)

    # Select protein from initial and save it as 'initial.gro'
    protein_initial = initial_universe.select_atoms("protein")
    protein_initial.write("initial.gro")

    # Select protein and lipids (PA, PC, OL) from target and save as 'target.gro'
    protein_and_lipids = target_universe.select_atoms(
        "protein or resname PA or resname PC or resname OL"
    )
    protein_and_lipids.write("target.gro")

    print("Files 'initial.gro' and 'target.gro' have been created.")


def prepare_memento_input(initial_gro, target_gro, run_name):
    # Current directory
    current_dir = os.getcwd()

    # Change to the folder path
    if not os.path.exists(run_name):
        os.makedirs(run_name)
        logging.info(f"Folder '{run_name}' created successfully.")
    else:
        logging.info(f"Folder '{run_name}' already exists.")

    # Create a folder called wdir in the run_name folder
    os.chdir(run_name)
    if not os.path.exists("wdir"):
        os.makedirs("wdir")
        logging.info("Folder 'wdir' created successfully.")
    else:
        logging.info("Folder 'wdir' already exists.")

    # Call the function to prepare input structures
    prepare_input_structures(initial_gro, target_gro)

    # Get to original folder
    os.chdir(current_dir)
