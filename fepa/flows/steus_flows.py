import os
import gromacs
import pandas as pd
import logging
import shutil
from pathlib import Path
from fepa.utils.wham_utils import (
    parse_us_plumed_file,
    process_wham_path,
    plot_free_combined,
    analyse_us_hist,
)
from fepa.utils.steus_utils import (
    write_temp_MDP,
    get_window_weights,
    setup_steus,
    extract_temp_at_marker,
    extract_time_at_marker,
    filter_colvar_pandas,
)
from fepa.utils.wham_utils import (
    create_colvar_chunks,
    generate_metadata,
    parse_us_plumed_file,
    plot_colvars,
    plot_histogram,
    process_colvars,
)
from multiprocessing import Pool


class steus_weight_sampling_workflow:
    """
    Class to handle sampling weight for steus simulations.
    """

    def __init__(
        self,
        sim_paths,
        temps=["310", "316", "322", "328", "334", "340", "346", "352", "358"],
        weights_sampling_folder="weight_sampling",
    ):
        self.sim_paths = sim_paths
        self.temps = temps
        self.weights_sampling_folder = weights_sampling_folder

    def setup_simulations(self, exist_ok=False):
        """
        Sets up the simulations for each sim path in sim_paths
        by creating a folder called weight_sampling with subfolders
        for each temperature in temps. It writes the MDP file
        for each temperature and runs grompp for each
        temperature.
        """
        for sim_path in self.sim_paths:
            cwd = os.getcwd()
            os.chdir(sim_path)

            for temp in self.temps:
                temp_folder = f"{self.weights_sampling_folder}/T{temp}"
                os.makedirs(temp_folder, exist_ok=exist_ok)
                logging.info(f"Creating {temp_folder}")
                write_temp_MDP(temp)
                logging.info(f"Writing MDP file for {temp}")
                gromacs.grompp(
                    f=f"T{temp}.mdp",
                    c="prod.gro",
                    r="prod.gro",
                    n="index.ndx",
                    p="topol.top",
                    o=f"{temp_folder}/T{temp}",
                )
            os.chdir(cwd)

    def run_simulations(self):
        """
        Runs the weight sampling simulations for each sim path
        in sim_paths for each temperature in temps.
        """
        for sim_path in self.sim_paths:
            cwd = os.getcwd()
            os.chdir(sim_path)

            for temp in self.temps:
                temp_folder = f"{self.weights_sampling_folder}/T{temp}"
                gromacs.mdrun(deffnm=f"{temp_folder}/T{temp}", v=True)
                logging.info(f"Finished MD simulation for {temp}")
            os.chdir(cwd)

    def get_weights_from_simulations(self):
        """
        Gets the weights from the simulations for each sim path
        in sim_paths
        """
        weight_lists = []  # List to store weight lists for each simulation; This is a list of lists
        path_strings = []
        for sim_path in self.sim_paths:
            weight_sampling_path = os.path.join(sim_path, self.weights_sampling_folder)
            weight_lists.append(get_window_weights(self.temps, weight_sampling_path))
            path_strings.append(weight_sampling_path)
        logging.info(f"Weight strings: {weight_lists}")
        logging.info(f"Weight sampling paths: {path_strings}")
        return weight_lists


class steus_umbrella_sampling_workflow:
    """
    Class to handle umbrella sampling for steus simulations.
    """

    def __init__(
        self,
        wdir_path,
        plumed_path,
        submission_script_template_arr,
        template_mdp,
        temp_weights,
        temp_lambdas,
        start,
        end,
        steus_folder_name="steus_v1",
        n_windows=24,
        wham_dirname="wham",
    ):
        logging.info("Initializing steus_umbrella_sampling_workflow...")
        self.wdir_path = wdir_path
        self.plumed_path = plumed_path
        self.submission_script_template_arr = submission_script_template_arr
        self.template_mdp = template_mdp
        self.temp_weights = temp_weights
        self.temp_lambdas = temp_lambdas
        self.steus_folder_name = steus_folder_name
        self.n_windows = n_windows
        self.steus_path = os.path.join(self.wdir_path, self.steus_folder_name)
        self.wham_path = os.path.join(self.wdir_path, self.steus_folder_name, wham_dirname)
        self.start = start
        self.end = end

    def setup_simulations(self, exist_ok=False):
        setup_steus(
            self.wdir_path,
            self.plumed_path,
            self.submission_script_template_arr,
            self.template_mdp,
            self.temp_weights,
            self.temp_lambdas,
            steus_name=self.steus_folder_name,
            exist_ok=exist_ok,
        )

    def extract_ground_state_colvar(self):
        logging.info("Extracting ground state colvar data...")
        # For each window
        for i in range(self.n_windows):
            logging.info(f"Processing simulation {i}")
            filename = f"{self.wdir_path}/{self.steus_folder_name}/sim{i}/sim{i}.log"  # Replace with the actual filename
            # Extract temperature and time values from the gmx log file
            temperatures = extract_temp_at_marker(filename)
            times = extract_time_at_marker(filename)

            logging.info(f"Extracted {len(temperatures)} temperature values.")
            logging.info(f"Extracted {len(times)} time values.")

            # Create a DataFrame from the extracted data
            df = pd.DataFrame({"Time (ps)": times, "Temperature (K)": temperatures})

            # Save the DataFrame to a CSV file
            temp_file_name = filename.replace(".log", "_temp.csv")
            df.to_csv(temp_file_name, index=False)
            logging.info(f"DataFrame saved to {temp_file_name}.")

            colvar_file = f"{self.wdir_path}/{self.steus_folder_name}/sim{i}/COLVAR"

            # Filter the colvar file for the ground state temperature
            filter_colvar_pandas(colvar_file, temp_file_name)

    def prepare_wham(self):
        # Create wham folder in steus path
        logging.info("Preparing WHAM folder...")
        os.makedirs(self.wham_path, exist_ok=True)
        logging.info(f"Created WHAM folder at: {self.wham_path}")
        colvar_path = os.path.join(self.wham_path, "colvars_100_pct_forward")
        os.makedirs(colvar_path, exist_ok=True)
        logging.info(f"Created COLVAR folder at: {colvar_path}")
        # Copy all the colvar_groundstate files from all sims folder in steus_path to colvar_path
        for i in range(0, self.n_windows):
            sim_path = os.path.join(self.steus_path, f"sim{i}")
            colvar_file = os.path.join(sim_path, "COLVAR_groundstate")
            destination_path = os.path.join(colvar_path, f"COLVAR_groundstate_{i}")
            if os.path.exists(colvar_file):
                shutil.copy(colvar_file, destination_path)
                logging.info(f"Copied {colvar_file} to {destination_path}")
        # Plot colvars
        plot_colvars(colvar_path)
        # Process colvars
        process_colvars(colvar_path=colvar_path, relaxation_time=500)
        # Plot processed colvars
        plot_histogram(colvar_path)
        kappa_values = []
        at_values = []
        # Get kappa and at values from plumed files
        for i in range(24):
            plumed_file = os.path.join(self.steus_path, f"sim{i}", "steus_plumed.dat")
            kappa, at = parse_us_plumed_file(plumed_file)
            kappa_values.append(kappa)
            at_values.append(at)
        logging.info(f"Kappa values: {kappa_values}")
        logging.info(f"AT values: {at_values}")
        # Ensure all kappa values are the same
        assert all(k == kappa_values[0] for k in kappa_values), (
            "Kappa values are not consistent."
        )
        generate_metadata(
            colvar_path,
            kappa_values[0],
            at_values,
            output_file="metadata.dat",
            decorrelation_time=0.02,
        )
        ## Process each chunk size in both forward and reverse directions
        for chunk_size in [20, 40, 60, 80]:
            print(f"Processing chunk size: {chunk_size}% for {colvar_path}")
            # Forward direction
            forward_chunk_path = create_colvar_chunks(
                wham_path=self.wham_path,
                colvar_100_pct_path=colvar_path,
                chunk_size_percentage=chunk_size,
                direction="forward",
            )
            # Call the plotting function
            plot_histogram(forward_chunk_path)
            # Create metadata file
            generate_metadata(
                forward_chunk_path,
                kappa_values[0],
                at_values,
                output_file="metadata.dat",
                decorrelation_time=0.02,
            )
            # Reverse direction
            reverse_chunk_path = create_colvar_chunks(
                wham_path=self.wham_path,
                colvar_100_pct_path=colvar_path,
                chunk_size_percentage=chunk_size,
                direction="reverse",
            )
            # Call the plotting function
            plot_histogram(reverse_chunk_path)
            # Create metadata file
            generate_metadata(
                reverse_chunk_path,
                kappa_values[0],
                at_values,
                output_file="metadata.dat",
                decorrelation_time=0.02,
            )

    def run_wham(self):
        if not os.path.exists(self.wham_path):
            # error
            raise FileNotFoundError(f"WHAM path does not exist: {self.wham_path}")
        # Create a list to store prepared WHAM paths
        prepared_wham_paths = []
        prepared_wham_paths.append(
            os.path.join(self.wham_path, "colvars_100_pct_forward")
        )
        for pct in ["20", "40", "60", "80"]:
            prepared_wham_paths.append(
                os.path.join(self.wham_path, f"colvars_{pct}_pct_forward")
            )
            prepared_wham_paths.append(
                os.path.join(self.wham_path, f"colvars_{pct}_pct_reverse")
            )
        # Print prepared WHAM paths
        logging.info(f"Prepared WHAM paths: {prepared_wham_paths}")

        # Get at values
        kappa_values = []
        at_values = []
        # Get kappa and at values from plumed files
        for i in range(self.n_windows):
            plumed_file = os.path.join(self.steus_path, f"sim{i}", "steus_plumed.dat")
            kappa, at = parse_us_plumed_file(plumed_file)
            kappa_values.append(kappa)
            at_values.append(at)

        # Use Pool to parallelize WHAM runs
        with Pool(processes=2) as pool:
            # Map each wham_path to the process_wham_path function
            pool.starmap(
                process_wham_path,
                [(wham_path, at_values) for wham_path in prepared_wham_paths],
            )

    def analyse_us_hist(self, range=(90, 180), colvar_filename="COLVAR"):
        # Make us_path
        us_path = os.path.join(self.wdir_path, self.steus_folder_name)

        # Analyse
        analyse_us_hist(
            us_path,
            colvar_filename=colvar_filename,
            label=f"{self.start}_{self.end}",
            range=range,
        )

    def plot_free_energies(
        self,
        units="kcal",
    ):
        plot_free_combined(
            self.wham_path,
            structure_1=self.start,
            structure_2=self.end,
            units=units,
            box_CV_means_csv=os.path.join(self.steus_path, "mean_values.csv"),
        )
