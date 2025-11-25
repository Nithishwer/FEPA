import os
import shutil
import pandas as pd
import logging
import re
import subprocess


from fepa.utils.reus_utils import (
    setup_reus,
)
from fepa.utils.plumed_utils import (
    add_resid_offset_to_ca_indices,
)
from fepa.utils.wham_utils import (
    process_wham_path,
    plot_free_combined,
    analyse_us_hist,
    create_colvar_chunks,
    generate_metadata,
    parse_hrex_us_plumed_file,
    plot_colvars,
    plot_histogram,
    process_colvars,
)
from multiprocessing import Pool


class reus_umbrella_sampling_workflow:
    """
    Class to handle umbrella sampling for reus simulations.
    """

    def __init__(
        self,
        wdir_path,
        plumed_path,
        submission_script_template_arr,
        start,
        end,
        initial_gro=None,
        target_gro=None,
        reus_folder_name="reus_v1",
        n_windows=24,
        plumed_resid_offset=None,
        plumed_resid_break=None,
    ):
        logging.info("Initializing reus_umbrella_sampling_workflow...")
        self.wdir_path = wdir_path
        self.plumed_path = plumed_path
        self.submission_script_template_arr = submission_script_template_arr
        self.reus_folder_name = reus_folder_name
        self.n_windows = n_windows
        self.reus_path = os.path.join(self.wdir_path, self.reus_folder_name)
        self.wham_path = os.path.join(self.wdir_path, self.reus_folder_name, "wham")
        self.start = start
        self.end = end
        self.plumed_resid_offset = plumed_resid_offset
        self.plumed_resid_break = plumed_resid_break
        self.initial_gro = initial_gro
        self.target_gro = target_gro

    def setup_simulations(self, exist_ok=False):
        setup_reus(
            self.wdir_path,
            self.plumed_path,
            sim_name=self.start + "_" + self.end,
            reus_name=self.reus_folder_name,
            job_template=self.submission_script_template_arr,
            n_windows=self.n_windows,
            plumed_resid_offset=self.plumed_resid_offset,
            plumed_resid_break=self.plumed_resid_break,
        )

    def prepare_wham(self, n_windows=24):
        # Create wham folder in reus path
        logging.info("Preparing WHAM folder...")
        os.makedirs(self.wham_path, exist_ok=True)
        logging.info(f"Created WHAM folder at: {self.wham_path}")
        colvar_path = os.path.join(self.wham_path, "colvars_100_pct_forward")
        os.makedirs(colvar_path, exist_ok=True)
        logging.info(f"Created COLVAR folder at: {colvar_path}")
        # Copy all the colvar_groundstate files from all sims folder in reus_path to colvar_path
        for i in range(0, self.n_windows):
            sim_path = os.path.join(self.reus_path, f"sim{i}")
            colvar_file = os.path.join(sim_path, f"COLVAR.{i}")
            destination_path = os.path.join(colvar_path, f"COLVAR_{i}")
            if os.path.exists(colvar_file):
                shutil.copy(colvar_file, destination_path)
                logging.info(f"Copied {colvar_file} to {destination_path}")
        # Plot colvars
        plot_colvars(colvar_path)
        # Process colvars
        process_colvars(colvar_path=colvar_path, relaxation_time=500)
        # Plot processed colvars
        plot_histogram(colvar_path)
        kappa, at_values = parse_hrex_us_plumed_file(self.plumed_path)
        logging.info(f"Kappa value: {kappa}")
        logging.info(f"AT values: {at_values}")

        generate_metadata(
            colvar_path,
            kappa,
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
                kappa,
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
                kappa,
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
        _, at_values = parse_hrex_us_plumed_file(self.plumed_path)

        # Use Pool to parallelize WHAM runs
        with Pool(processes=2) as pool:
            # Map each wham_path to the process_wham_path function
            pool.starmap(
                process_wham_path,
                [(wham_path, at_values) for wham_path in prepared_wham_paths],
            )

    def analyse_us_hist(
        self, range=(90, 180), colvar_filename=None, colvar_prefix=None
    ):
        # Make us_path
        us_path = os.path.join(self.wdir_path, self.reus_folder_name)

        if colvar_filename is not None:
            # Analyse
            analyse_us_hist(
                us_path,
                colvar_prefix=colvar_filename,
                label=f"{self.start}_{self.end}",
                range=range,
            )
        elif colvar_prefix is not None:
            # Analyse
            analyse_us_hist(
                us_path,
                colvar_prefix=colvar_prefix,
                label=f"{self.start}_{self.end}",
                range=range,
            )
        else:
            # Error
            raise ValueError(
                "Either colvar_filename or colvar_prefix must be provided."
            )

    def plot_free_energies(
        self,
        units="kcal",
    ):
        plot_free_combined(
            self.wham_path,
            structure_1=self.start,
            structure_2=self.end,
            structure_1_CV=self.initial_CV,
            structure_2_CV=self.target_CV,
            units=units,
        )

    def get_initial_final_CVs(self):
        """
        Get the initial and final CV values from the plumed file.
        """
        # Create a folder in wdir for initial and final gro files
        logging.info("Creating initial and target folders...")
        initial_folder = os.path.join(self.wdir_path, "initial_structure")
        target_folder = os.path.join(self.wdir_path, "target_structure")
        os.makedirs(initial_folder, exist_ok=True)
        os.makedirs(target_folder, exist_ok=True)
        # Get the initial and final gro files
        initial_gro_file = os.path.join(initial_folder, "initial.gro")
        target_gro_file = os.path.join(target_folder, "target.gro")
        # Copy the initial and final gro files to the respective folders
        logging.info(
            f"Copying initial gro file ({self.initial_gro}) to {initial_gro_file}..."
        )
        shutil.copy(self.initial_gro, initial_gro_file)
        # Copy the target gro file to the respective folder
        logging.info(
            f"Copying target gro file ({self.target_gro}) to {target_gro_file}..."
        )
        shutil.copy(self.target_gro, target_gro_file)
        # Copy the plumed file to the respective folders
        logging.info("Copying plumed files")
        shutil.copy(self.plumed_path, os.path.join(initial_folder, "plumed.dat"))
        shutil.copy(self.plumed_path, os.path.join(target_folder, "plumed.dat"))

        if self.plumed_resid_offset is not None:
            logging.info("Adding residue offset to CA indices in plumed file...")
            add_resid_offset_to_ca_indices(
                self.plumed_path,
                os.path.join(initial_folder, "plumed.dat"),
                self.plumed_resid_offset,
                self.plumed_resid_break,
            )
            add_resid_offset_to_ca_indices(
                self.plumed_path,
                os.path.join(target_folder, "plumed.dat"),
                self.plumed_resid_offset,
                self.plumed_resid_break,
            )
        # Replace the reference.pdb path in the plumed file
        logging.info("Replacing reference.pdb path in plumed file...")
        with open(os.path.join(initial_folder, "plumed.dat"), "r") as f:
            plumed_file = f.read()
        plumed_file = re.sub(r"\.\./reference.pdb", "reference.pdb", plumed_file)
        with open(os.path.join(initial_folder, "plumed.dat"), "w") as f:
            f.write(plumed_file)
        with open(os.path.join(target_folder, "plumed.dat"), "r") as f:
            plumed_file = f.read()
        plumed_file = re.sub(r"\.\./reference.pdb", "reference.pdb", plumed_file)
        with open(os.path.join(target_folder, "plumed.dat"), "w") as f:
            f.write(plumed_file)
        # Copy reference pdb file to the respective folders
        logging.info("Copying reference.pdb file to initial and target folders...")
        reference_pdb = os.path.join(self.reus_path, "reference.pdb")
        shutil.copy(reference_pdb, os.path.join(initial_folder, "reference.pdb"))
        shutil.copy(reference_pdb, os.path.join(target_folder, "reference.pdb"))
        # Run the plumed driver on the initial and final gro files
        logging.info("Running plumed driver...")
        plumed_driver_command = "plumed driver --plumed plumed.dat --igro {gro_file}"
        # cd into the intial and target folders and run the plumed driver
        cwd = os.getcwd()
        os.chdir(initial_folder)
        subprocess.run(
            plumed_driver_command.format(gro_file=initial_gro_file), shell=True
        )
        os.chdir(target_folder)
        subprocess.run(
            plumed_driver_command.format(gro_file=target_gro_file), shell=True
        )
        # Read COLVAR file in initial folder to get initial CV value
        logging.info("Reading COLVAR files to get initial and final CV values...")
        initial_colvar_file = os.path.join(initial_folder, "COLVAR")
        initial_colvar_df = pd.read_csv(
            initial_colvar_file, delim_whitespace=True, skiprows=1, names=["time", "CV"]
        )
        self.initial_CV = initial_colvar_df["CV"].values[0]
        logging.info(f"Initial CV value: {self.initial_CV}")
        # Read COLVAR file in target folder to get final CV value
        logging.info(f"Reading COLVAR file in target folder: {target_folder}")
        target_colvar_file = os.path.join(target_folder, "COLVAR")
        target_colvar_df = pd.read_csv(
            target_colvar_file, delim_whitespace=True, skiprows=1, names=["time", "CV"]
        )
        self.target_CV = target_colvar_df["CV"].values[0]
        logging.info(f"Target CV value: {self.target_CV}")
        os.chdir(cwd)
