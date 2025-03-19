import os
import shutil
import pandas as pd
import gromacs
import re
import numpy as np
import subprocess

print(gromacs.release())


def setup_reus_v2(wdir_path, plumed_path_template):
    # Navigate to memento_dir/wdir/boxes
    sim_name = os.path.basename(os.path.dirname(os.path.dirname(wdir_path)))
    print(f"Setting up REUS for the complex : {sim_name}.")

    # Define the path to the 'boxes' folder
    boxes_path = os.path.join(wdir_path, "boxes")

    # Create 'reus' folder inside the working directory if it doesn't exist
    reus_path = os.path.join(wdir_path, "reus_v2")
    os.makedirs(reus_path, exist_ok=True)

    # Get the list of folder names inside 'boxes'
    box_folders = ["sim" + str(i) for i in range(24)]

    # Copy plumed file
    plumed_path = plumed_path_template.replace("$COMPOUND", sim_name[4:])
    shutil.copy(plumed_path, os.path.join(reus_path, "plumed.dat"))
    shutil.copy(
        "/biggin/b211/reub0138/Projects/orexin/lenselink_a2a_memento_v1/job_archer_reus.sh",
        reus_path,
    )

    # Replace all occurences of jobname in job_archer_reus.sh with sim_name
    with open(
        os.path.join(reus_path, os.path.join(reus_path, "job_archer_reus.sh")), "r"
    ) as f:
        lines = f.readlines()
    with open(
        os.path.join(reus_path, os.path.join(reus_path, "job_archer_reus.sh")), "w"
    ) as f:
        for line in lines:
            f.write(line.replace("JOBNAME", sim_name[4:]))

    # Iterate through each folder in 'boxes' and replicate the folder structure in 'reus'
    for folder in box_folders:
        # Define the new folder path inside 'reus'
        new_folder_path = os.path.join(reus_path, folder)
        os.makedirs(new_folder_path, exist_ok=True)

        # Paths to the folders to be copied
        toppar_source = os.path.join(boxes_path, folder, "toppar")
        amber_source = os.path.join(boxes_path, folder, "amber99sb-star-ildn-mut.ff")

        # Copy 'toppar' and 'amber99sb-star-ildn-mut.ff' if they exist
        if os.path.exists(toppar_source):
            shutil.copytree(
                toppar_source,
                os.path.join(new_folder_path, "toppar"),
                dirs_exist_ok=True,
            )
        if os.path.exists(amber_source):
            shutil.copytree(
                amber_source,
                os.path.join(new_folder_path, "amber99sb-star-ildn-mut.ff"),
                dirs_exist_ok=True,
            )

        # copy other inout files
        shutil.copy(os.path.join(boxes_path, folder, "index.ndx"), new_folder_path)
        shutil.copy(
            os.path.join(boxes_path, folder, "prod.gro"),
            os.path.join(new_folder_path, "equilibrated.gro"),
        )
        shutil.copy(
            os.path.join(boxes_path, folder, "topol.top"),
            os.path.join(new_folder_path, "topol.top"),
        )
        # Copy template mdps
        shutil.copy(
            os.path.join(boxes_path, folder, "prod.mdp"),
            os.path.join(new_folder_path, "prod.mdp"),
        )
        # Change nsteps to 100ns
        with open(os.path.join(new_folder_path, "prod.mdp"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(new_folder_path, "prod.mdp"), "w") as f:
            for line in lines:
                if line.startswith("nsteps"):
                    f.write("nsteps = 50000000\n")  # Change nsteps to 100ns
                elif line.startswith("define"):
                    continue  # remove posres
                else:
                    f.write(line)

    # make reference from equilibrated.gro
    subprocess.run(
        f"gmx editconf -f {os.path.join(new_folder_path, 'equilibrated.gro')} -o {os.path.join(reus_path, 'reference.pdb')} ",
        shell=True,
    )

    # Success
    print(f"'reus' folder structure created successfully at {reus_path}")


def main():
    # PLumed template file
    plumed_path_template = f"/biggin/b211/reub0138/Projects/orexin/lenselink_a2a_memento_v1/binding_pocket_analysis/PCA_v3_plumed_CV_v1_output/$COMPOUND_apo_r1/plumed.dat"
    # Example: You can replace the input path with a dynamic input mechanism if needed.
    complex_list = [
        "A2A_cmp_1",
        "A2A_cmp_2",
        "A2A_cmp_3",
        "A2A_cmp_8",
        "A2A_cmp_11",
        "A2A_cmp_13",
        "A2A_cmp_17",
    ]
    for complex in complex_list:
        wdir = f"/biggin/b211/reub0138/Projects/orexin/lenselink_a2a_memento_v1/{complex}/memento_run_v0/wdir"
        setup_reus_v2(wdir, plumed_path_template)


# Ensure the script runs only when executed as the main program
if __name__ == "__main__":
    main()
