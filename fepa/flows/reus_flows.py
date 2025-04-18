import os
import shutil
import pandas as pd
import logging
import re
import numpy as np
import subprocess


from fepa.utils.reus_utils import (
    setup_reus,
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
        reus_folder_name="reus_v1",
        n_windows=24,
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

    def setup_simulations(self, exist_ok=False):
        setup_reus(
            self.wdir_path,
            self.plumed_path,
            sim_name=self.start + "_" + self.end,
            reus_name=self.reus_folder_name,
            job_template=self.submission_script_template_arr,
            n_windows=self.n_windows,
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

    def analyse_us_hist(self, range=(90, 180), colvar_filename=None, colvar_prefix=None):
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
            units=units,
            box_CV_means_csv=os.path.join(self.reus_path, "mean_values.csv"),
        )
