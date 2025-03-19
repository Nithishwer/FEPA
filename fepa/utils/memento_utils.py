import os
import logging
import MDAnalysis as mda
from PyMEMENTO import MEMENTO


def run_pymemento(last_run=None, template_path=None, protonation_states=None):
    if template_path is None:
        logging.error("Template path is not defined.")
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
            range(1, 297)
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
