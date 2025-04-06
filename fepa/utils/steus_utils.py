import pandas as pd
import logging
import shutil
import gromacs
import pyedr
import os


def make_test_mdp():
    mdp = """
    integrator              = md
    dt                      = 0.002
    nsteps                  = 250000  ; 500 ps
    nstxout-compressed      = 5000
    nstxout                 = 0
    nstvout                 = 0
    nstfout                 = 0
    nstcalcenergy           = 50
    nstenergy               = 50
    nstlog                  = 5000
    ;
    cutoff-scheme           = Verlet
    nstlist                 = 20
    rlist                   = 0.9
    vdwtype                 = Cut-off
    vdw-modifier            = None
    DispCorr                = EnerPres
    rvdw                    = 0.9
    coulombtype             = PME
    rcoulomb                = 0.9
    ;
    tcoupl                  = v-rescale
    tc_grps                 = POPC_FIP SOLV
    tau_t                   = 1.0 1.0
    ref_t                   = {T} {T}
    ;
    pcoupl                  = C-rescale
    pcoupltype              = semiisotropic 
    tau_p                   = 5.0
    compressibility         = 4.5e-5  4.5e-5
    ref_p                   = 1.0     1.0
    refcoord_scaling        = com
    ;
    constraints             = h-bonds
    constraint_algorithm    = LINCS
    continuation            = no
    gen-vel                 = yes
    gen-temp                = {T}
    ;
    nstcomm                 = 100
    comm_mode               = linear
    comm_grps               = POPC_FIP SOLV

    ; Pull code
    pull                     = yes
    pull_ncoords             = 1
    pull_ngroups             = 2
    pull_group1_name         = FIP
    pull_group2_name         = POPC
    pull_group2_pbcatom      = 995
    pull_pbc_ref_prev_step_com = yes
    pull_coord1_type         = umbrella
    pull_coord1_geometry     = direction
    pull_coord1_vec          = 0 0 1
    pull_coord1_dim          = N N Y  ; pull along z
    pull_coord1_groups       = 1 2
    pull_coord1_start        = yes
    pull_coord1_rate         = 0.0   ; restrain in place
    pull_coord1_k            = 1000  ; kJ mol^-1 nm^-2
    pull_nstfout             = 50

    """

    # These sometimes very small steps are tested because I might want to run STeUS
    # either with delta T of 4 or of 5, and I'll need these simulations to calculate
    # the weights, so I might as well run them all now while testing stability.
    temps = ["310", "316", "322", "328", "334", "340", "346", "352", "358"]

    for temp in temps:
        with open(f"T{temp}.mdp", "w") as f:
            f.write(mdp.format(T=temp))
        os.system(
            f"gmx grompp -f T{temp}.mdp -c ../../window_prep/alchembed13.gro -r ../../window_prep/alchembed13.gro -n ../../window_prep/index.ndx -p ../../window_prep/topol.top -o runs/T{temp}"
        )


def filter_colvar_by_temp(colvar_file, temp_time_file, T=310.0):
    """
    Reads a colvar file and a temp_time.csv file using pandas,
    filters the colvar file based on the integer part of the time column
    and the temperature in the temp_time file, and saves the filtered
    data to a new colvar file.

    Args:
        colvar_file (str): Path to the input colvar file.
        temp_time_file (str): Path to the input temp_time.csv file.
        output_colvar_file (str): Path to the output filtered colvar file.
        T (float): Temperature to filter the temp_time file. Default is 310.0.
    """

    # Read temp_time.csv using pandas
    temp_df = pd.read_csv(temp_time_file)
    # Filter temp_time DataFrame for Temperature (K) == T
    filtered_temp_df = temp_df[temp_df["Temperature (K)"] == T]
    # Get the set of integer times from the filtered temp_time DataFrame
    valid_times = set(filtered_temp_df["Time (ps)"].astype(int))
    # Read the colvar file using pandas
    colvar_df = pd.read_csv(colvar_file, comment="#", sep="\s+", names=["time", "CV"])
    # Filter the colvar DataFrame
    filtered_colvar_df = colvar_df[colvar_df["time"].astype(int).isin(valid_times)]

    # outfile name
    output_colvar_file = colvar_file + ("_groundstate")
    # Save the filtered colvar DataFrame to a new file
    with open(output_colvar_file, "w") as outfile:
        # Write the header
        with open(colvar_file, "r") as infile:
            for line in infile:
                if line.startswith("#"):
                    outfile.write(line)
                    break
        # Write the data
        filtered_colvar_df.to_csv(outfile, sep=" ", index=False, header=False)

    logging.info(f"Filtered colvar data saved to {output_colvar_file}")


def filter_colvar_pandas(colvar_file, temp_time_file):
    """
    Reads a colvar file and a temp_time.csv file using pandas,
    filters the colvar file based on the integer part of the time column
    and the temperature in the temp_time file, and saves the filtered
    data to a new colvar file.

    Args:
        colvar_file (str): Path to the input colvar file.
        temp_time_file (str): Path to the input temp_time.csv file.
        output_colvar_file (str): Path to the output filtered colvar file.
    """

    # Read temp_time.csv using pandas
    temp_df = pd.read_csv(temp_time_file)
    # Filter temp_time DataFrame for Temperature (K) == 310.0
    filtered_temp_df = temp_df[temp_df["Temperature (K)"] == 310.0]
    # Get the set of integer times from the filtered temp_time DataFrame
    valid_times = set(filtered_temp_df["Time (ps)"].astype(int))
    # Read the colvar file using pandas
    colvar_df = pd.read_csv(colvar_file, comment="#", sep="\s+", names=["time", "CV"])
    # Filter the colvar DataFrame
    filtered_colvar_df = colvar_df[colvar_df["time"].astype(int).isin(valid_times)]

    # outfile name
    output_colvar_file = colvar_file + ("_groundstate")
    # Save the filtered colvar DataFrame to a new file
    with open(output_colvar_file, "w") as outfile:
        # Write the header
        with open(colvar_file, "r") as infile:
            for line in infile:
                if line.startswith("#"):
                    outfile.write(line)
                    break
        # Write the data
        filtered_colvar_df.to_csv(outfile, sep=" ", index=False, header=False)

    logging.info(f"Filtered colvar data saved to {output_colvar_file}")


def extract_temp_at_marker(filename):
    """
    Extracts the temperature value from the line preceding the "<<" marker
    for each time step in the given gmx log file.

    Args:
        filename (str): The path to the gmx log file.

    Returns:
        list: A list of temperature values (floats) corresponding to each
              occurrence of the "<<" marker.
    """

    temperatures = []
    with open(filename, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "<<" in line:
            # Extract the temperature from the line before the marker
            temp = float(line.split()[1])  # Assuming Temp is the 2nd value
            temperatures.append(temp)
    return temperatures


def extract_time_at_marker(filename):
    """
    Extracts the time value before the line with the "<<" marker
    for each time step in the given gromacs log file.

    Args:
        filename (str): The path to the gmx log file.

    Returns:
        list: A list of time values (floats) corresponding to each
              occurrence of the "<<" marker.
    """

    times = []
    with open(filename, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "<<" in line:
            # Look in reverse for a line with 'Time'
            for j in range(i - 1, -1, -1):
                if "Time" in lines[j]:
                    # Extract the time value
                    time = lines[j + 1].split()[
                        1
                    ]  # Assuming Time is the 2nd value in the line following 'Time'
                    times.append(time)
                    break
    return times


def write_temp_MDP(temp):
    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # Get the directory of the script
    template_path = os.path.join(
        script_dir, "../data/template_steus_weight_sampling.mdp"
    )
    with open(template_path, "r") as template_file:
        sample_mdp = template_file.read()
    with open(f"T{temp}.mdp", "w") as f:
        f.write(sample_mdp.format(T=temp))


def relative_weight(i, betas, e_pots):
    """
    Code from Bjarne Feddersen
    Assumes that i is the index of window n, and j is the index
    of window n + 1. Returns the relative weight g(n+1) - g(n) as in eq. 6
    of Sousa et al. https://doi.org/10.1021/acs.jctc.2c01162
    """
    j = i + 1
    delta_beta = betas[j] - betas[i]
    avg_e_pot = (e_pots[i] + e_pots[j]) / 2
    return delta_beta * avg_e_pot


def get_window_weights(temps, weight_sampling_path):
    """
    Code from Bjarne Feddersen

    Get the window weights for the steus simulations from weight sampling
    Args:
        temps (list): List of temperatures
        weight_sampling_path (str): Path to the weight sampling folder
    Returns:
        str: String of weights that can be copied into the mdp file
    """
    e_pots = []
    R = 0.0083144621
    betas = [1 / (R * int(T)) for T in temps]

    for temp in temps:
        edr_path = os.path.join(weight_sampling_path, f"T{temp}", f"T{temp}.edr")
        edr = pyedr.edr_to_dict(edr_path)
        avg_e_pot = edr["Potential"].mean()
        e_pots.append(avg_e_pot)

    rel_weights = [relative_weight(i, betas, e_pots) for i in range(len(e_pots) - 1)]

    # first weight initially assumed to be 0
    window_weights = [0]

    # for each delta weight, add the delta to the last absolute weight to obtain
    # the next absolute weight
    for rel_weight in rel_weights:
        window_weights.append(window_weights[-1] + rel_weight)

    # subtract the average weight from each of the absolute weights
    # to make the sum of the weights be equal to 0
    # as seen in http://dx.doi.org/10.1103/PhysRevE.76.016703 below table 1
    avg_weight = sum(window_weights) / len(window_weights)
    window_weights = [weight - avg_weight for weight in window_weights]

    weight_list = [(round(weight)) for weight in window_weights]
    # write out string of weights that can be copied into the mdp file
    return weight_list


def add_steus_code_to_mdp(mdp_path, temp_weights, temp_lambdas):
    steus_code = """\n
;----------------------------------------------------
; EE/STEUS Stuff
;----------------------------------------------------
free-energy              = expanded
nstexpanded              = 500  ; attempt temperature change every 500 simulation steps
lmc-stats                = wang-landau  ; update expanded ensemble weights with wang-landau algorithm
lmc-move                 = metropolis
init-wl-delta            = 1 ; initial value of the Wang-Landay incrementor in kT
init-lambda-state        = 0
init-lambda-weights      = {temp_weights} ; vector of the initial weights (free energies in kT) used for the expanded ensemble states. Vector of floats, length must match lambda vector lengths
lmc-weights-equil        = wl-delta
weight-equil-wl-delta    = 0.0001
wl-ratio                 = 0.7
wl-scale                 = 0.8
wl-oneovert              = yes
simulated-tempering      = yes
sim-temp-low             = 310
sim-temp-high            = 358
temperature-lambdas      = {temp_lambdas}
simulated-tempering-scaling = linear  ; linearly interpolates the temperatures using the values of temperature-lambdas
"""

    # Read the existing file content
    with open(mdp_path, "r") as f:
        lines = f.readlines()

    # Find the first line that starts with 'free-energy' and truncate everything after it
    new_lines = []
    found = False
    for line in lines:
        if not found and line.strip().startswith("free-energy"):
            found = True
        if found:
            break
        new_lines.append(line)

    # Write the truncated content back to the file
    with open(mdp_path, "w") as f:
        f.writelines(new_lines)

    # Add the steus code to the end of the mdp file
    with open(mdp_path, "a") as f:
        f.write(
            steus_code.format(
                temp_weights=" ".join([str(weight) for weight in temp_weights]),
                temp_lambdas=" ".join(temp_lambdas),
            )
        )


def make_steus_plumed_dat(reus_plumed_dat, steus_plumed_dat, i):
    """
    Creates a modified version of the input file with only the i-th restraint value.

    Args:
        reus_plumed_dat (str): Path to the input file
        steus_plumed_dat (str): Path where to save the modified file
        i (int): Index of the restraint value to keep (0-based)

    Returns:
        str: Path to the created output file
    """
    with open(reus_plumed_dat, "r") as f:
        content = f.readlines()

    # Find and modify the restraint line
    for idx, line in enumerate(content):
        if line.strip().startswith("MOLINFO"):
            parts = line.split()
            for j, part in enumerate(parts):
                if part == "STRUCTURE=../reference.pdb":
                    # Modify to keep only the i-th value
                    parts[j] = "STRUCTURE=reference.pdb"
                    content[idx] = " ".join(parts) + "\n"
        if line.strip().startswith("restraint: RESTRAINT"):
            parts = line.split()

            for j, part in enumerate(parts):
                if part.startswith("AT=@replicas:"):
                    # Extract and validate values
                    values_str = part[len("AT=@replicas:") :]
                    values = values_str.split(",")

                    if i >= len(values):
                        raise ValueError(
                            f"Index {i} out of range. File has {len(values)} restraint values."
                        )

                    # Modify to keep only the i-th value
                    parts[j] = f"AT={values[i]}"
                    content[idx] = " ".join(parts) + "\n"
                    break
            break

    # Write the modified content
    with open(steus_plumed_dat, "w", encoding="UTF-8") as f:
        f.writelines(content)

    return values


def setup_steus(
    wdir_path,
    plumed_path,
    submission_script_template_arr,
    template_mdp,
    temp_weights,
    temp_lambdas,
    steus_name="steus_v1",
    exist_ok=False,
):
    # Get name of parent dir of parent dir or wdir
    run_name = os.path.basename(os.path.dirname(os.path.dirname(wdir_path)))

    logging.info(f"Setting up STEUS for : {run_name}")

    # Define the path to the 'boxes' folder
    boxes_path = os.path.join(wdir_path, "boxes")

    # Create 'steus' folder inside the working directory if it doesn't exist
    steus_path = os.path.join(wdir_path, steus_name)
    os.makedirs(steus_path, exist_ok=exist_ok)

    # Get the list of folder names inside 'boxes'
    sim_folders = [item for item in os.listdir(boxes_path) if item.startswith("sim")]

    # Copy the array job script template
    shutil.copy(
        submission_script_template_arr,
        os.path.join(steus_path, "job_ranv_steus_arr.sh"),
    )

    # Change job name in submission scripts
    with open(
        os.path.join(steus_path, "job_ranv_steus_arr.sh"),
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.read()
    with open(
        os.path.join(steus_path, "job_ranv_steus_arr.sh"),
        "w",
        encoding="utf-8",
    ) as f:
        new_lines = lines.replace("$JOBNAME", run_name)
        f.write(new_lines)

    # Iterate through each folder in 'boxes' and replicate the folder structure in 'steus'
    for sim_folder in sim_folders:
        logging.info(f"Setting up STEUS for : {sim_folder}")
        # Define the new folder path inside 'steus'
        steus_sim_folder_path = os.path.join(steus_path, sim_folder)
        os.makedirs(steus_sim_folder_path, exist_ok=True)

        # Paths to the folders to be copied
        toppar_source = os.path.join(boxes_path, sim_folder, "toppar")
        amber_source = os.path.join(
            boxes_path, sim_folder, "amber99sb-star-ildn-mut.ff"
        )

        # Copy 'toppar' and 'amber99sb-star-ildn-mut.ff' if they exist
        if os.path.exists(toppar_source):
            shutil.copytree(
                toppar_source,
                os.path.join(steus_sim_folder_path, "toppar"),
                dirs_exist_ok=True,
            )
        if os.path.exists(amber_source):
            shutil.copytree(
                amber_source,
                os.path.join(steus_sim_folder_path, "amber99sb-star-ildn-mut.ff"),
                dirs_exist_ok=True,
            )

        # copy other inout files
        shutil.copy(
            os.path.join(boxes_path, sim_folder, "index.ndx"), steus_sim_folder_path
        )
        shutil.copy(
            os.path.join(boxes_path, sim_folder, "prod.gro"),
            os.path.join(steus_sim_folder_path, "equilibrated.gro"),
        )
        shutil.copy(
            os.path.join(boxes_path, sim_folder, "topol.top"),
            os.path.join(steus_sim_folder_path, "topol.top"),
        )
        # Copy template mdps
        shutil.copy(
            template_mdp,
            os.path.join(steus_sim_folder_path, "prod.mdp"),
        )

        # Make steus plumed.dat
        make_steus_plumed_dat(
            plumed_path,
            os.path.join(steus_sim_folder_path, "steus_plumed.dat"),
            int(sim_folder.split("sim")[-1]),
        )

        logging.info(f"temp_weights: {temp_weights}")
        logging.info(f"temp_lambdas: {temp_lambdas}")

        # Add steus code to mdp
        add_steus_code_to_mdp(
            os.path.join(steus_sim_folder_path, "prod.mdp"),
            temp_weights=temp_weights,
            temp_lambdas=temp_lambdas,
        )

        # Make reference from equilibrated.gro
        gromacs.editconf(
            f=os.path.join(steus_sim_folder_path, "equilibrated.gro"),
            o=os.path.join(steus_sim_folder_path, "reference.pdb"),
        )

        # Make tpr file

    # Success
    print(f"'steus' folder structure created successfully at {steus_path}")
