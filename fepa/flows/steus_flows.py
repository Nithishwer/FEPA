import os
import gromacs
import pandas as pd
import logging
from fepa.utils.steus_utils import (
    write_temp_MDP,
    get_window_weights,
    setup_steus,
    extract_temp_at_marker,
    extract_time_at_marker,
    filter_colvar_pandas,
)


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
        steus_folder_name="steus_v1",
        n_windows=24,
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

    def analyse_simulations(self):
        """
        Analyse the simulations for each sim path in sim_paths
        by creating a folder called weight_sampling with subfolders
        for each temperature in temps. It writes the MDP file
        for each temperature and runs grompp for each
        temperature.
        """
        pass
