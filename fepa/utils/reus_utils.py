import os
import shutil
import subprocess
from fepa.utils.plumed_utils import add_resid_offset_to_ca_indices


def setup_reus(
    wdir_path,
    plumed_path,
    sim_name,
    job_template,
    reus_name="reus_v1",
    n_windows=24,
    plumed_resid_offset=None,
    plumed_resid_break=None,
):
    # Navigate to memento_dir/wdir/boxes
    print(f"Setting up REUS for the complex : {sim_name}.")

    # Define the path to the 'boxes' folder
    boxes_path = os.path.join(wdir_path, "boxes")

    # Create 'reus' folder inside the working directory if it doesn't exist
    reus_path = os.path.join(wdir_path, reus_name)
    os.makedirs(reus_path, exist_ok=True)

    # Get the list of folder names inside 'boxes'
    box_folders = ["sim" + str(i) for i in range(n_windows)]

    # Copy plumed file
    if plumed_resid_offset is not None:
        add_resid_offset_to_ca_indices(
            plumed_path,
            os.path.join(reus_path, "plumed.dat"),
            plumed_resid_offset,
            plumed_resid_break,
        )
    else:
        shutil.copy(plumed_path, os.path.join(reus_path, "plumed.dat"))
    shutil.copy(job_template, reus_path)

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
