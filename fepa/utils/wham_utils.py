import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import logging
import subprocess
import numpy as np


def read_cv_from_colvar(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        # Skip the first line and extract data
        data = [line.strip().split() for line in lines[1:]]
        # print(f"Number of rows in data of {filename}: {len(data)}")
        # print('Head of the data:', data[:5])

        # Filter out rows that contain non-numeric values (like '-')
        cleaned_data = [row for row in data if len(row) == 2]
        # print(f' Cleaned data of {filename}: {cleaned_data[:5]}')
        # print(f"Number of rows after cleaning: {len(cleaned_data)}")

        # Convert the cleaned data to a DataFrame
        df = pd.DataFrame(cleaned_data, columns=["time", "CV"], dtype=float)
        # print(df)
        # print('Head of the dataframe:', df.head())
        # print(df['CV'].values)
        return df["CV"].values


# Function to plot the histogram of the CV values from all files, with color differentiation
def plot_histogram(colvar_path):
    print(f"Plotting histogram of CV values for {colvar_path}...")
    plt.figure(figsize=(10, 6))

    # Iterate over all files in the directory that start with 'colvar'
    for filename in os.listdir(colvar_path):
        # if it has substring colvar
        if "processed_COLVAR" in filename:
            filepath = os.path.join(colvar_path, filename)
            print(f"Processing file: {filename}")

            cv_column = read_cv_from_colvar(filepath)
            # Plot histogram for this file's CV column, with a label for the legend
            # print(f"Number of CV values: {len(cv_column)}")
            plt.hist(cv_column, bins=50, alpha=0.5, label=filename, edgecolor="black")

    # Add labels and legend
    plt.title("Histogram of CV values by file")
    plt.xlabel("CV")
    plt.ylabel("Frequency")
    plt.legend(title="File")
    plt.savefig(colvar_path + "/histogram.png")
    plt.close()
    # plt.show()


def process_colvars(
    colvar_path, relaxation_time, colvar_prefix="COLVAR", max_lines=None
):
    """
    Processes the colvars data files by removing frames before relaxation time
    and after max lines then removes middle column then renames the files to processed_COLVAR.N.

    Args:
        colvar_path (str): Path to the directory containing colvars data files.
        relaxation_time (float): Relaxation time in ps.
        max_lines (int): Maximum number of lines to keep in each file (default: None).
    """
    # Get the current working directory
    cwd = os.getcwd()

    # Create the output directory if it doesn't exist
    os.chdir(colvar_path)

    # Raise error if no files starting with colvar are found
    if not glob.glob(colvar_prefix + "*"):
        raise FileNotFoundError(
            "Error: No files starting with 'colvar.' found in the directory."
        )

    # Loop over all files starting with 'COLVAR'
    for file in glob.glob(colvar_prefix + "*"):
        # Print
        print(f"Processing file: {file}")

        # Read the file
        df = pd.read_csv(
            file, sep="\s+", comment="#", header=None, names=["time", "CV"]
        )

        # Filter the DataFrame to keep only rows where 'time' is greater than or equal to 'min_time'
        df = df[df["time"] >= relaxation_time]

        if max_lines:
            # Keep only rows where the index is less than or equal to max_row_number
            df = df.iloc[: max_lines + 1]  # +1 to include the specified row number

        # Save the DataFrame with whitespace as the separator, no header, and no index
        with open(file, "w") as file_io:
            file_io.write("#! FIELDS time CV\n")
            # Use file_io instead of file in df.to_string
            df.to_string(file_io, index=False, header=False, float_format="%.6f")

        # Rename the file to processed_colvar.N
        new_file = file.replace("COLVAR", "processed_COLVAR")
        os.rename(file, new_file)

    os.chdir(cwd)  # Change back to the original directory


def analyse_us_hist(us_path, range, colvar_filename="COLVAR", label="steus_v1"):
    # Navigate to memento_dir/wdir/boxes
    logging.info(f"Analysing US histograms in {us_path}")

    # Get the list of folder names inside 'reus'
    us_folders = [
        f
        for f in os.listdir(us_path)
        if os.path.isdir(os.path.join(us_path, f)) and f.startswith("sim")
    ]
    logging.info(f"Found folders: {us_folders}")

    # Initialize a list to store folder names and corresponding means
    mean_values = []

    # Create a figure for the histograms
    plt.figure(figsize=(20, 12))

    # Loop through each folder
    for folder in us_folders:
        logging.info(f"Processing folder: {folder}")

        # Construct the path to the colvar file
        colvar_path = os.path.join(
            us_path, folder, colvar_filename
        )  # Path to colvar file

        # Read the colvar file, skipping the first line with '#! FIELDS time CV'
        data = pd.read_csv(colvar_path, sep="\s+", skiprows=1, names=["time", "CV"])

        # Calculate the mean of 'CV'
        mean_a_eig_1 = data["CV"].mean()

        # Append folder name and mean to the list
        mean_values.append({"folder": folder, "mean": mean_a_eig_1})

        # Plot the histogram for 'CV' on the same figure with some transparency and style
        plt.hist(data["CV"], bins=400, alpha=0.6, label=folder, range=range)

    # Use the basename of us_path as the plot title
    plot_title = label
    plt.title(f"Histograms of CV Across Folders - {plot_title}", fontsize=16)

    # Add labels for axes
    plt.xlabel("CV", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Add a legend with a nice font size
    plt.legend(title="Folders", fontsize=10, title_fontsize="12")

    # Add grid lines to the plot
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the combined histogram plot
    plt.savefig(os.path.join(us_path, f"{plot_title}_combined_histogram.png"), dpi=300)
    logging.info(
        f"Combined histogram saved as {os.path.join(us_path, f'{plot_title}_combined_histogram.png')}"
    )
    plt.close()

    # Save the mean values to a CSV file
    mean_df = pd.DataFrame(mean_values)
    mean_df.to_csv(os.path.join(us_path, "mean_values.csv"), index=False)

    logging.info(f"Mean values saved to {os.path.join(us_path, 'mean_values.csv')}")


def parse_steus_plumed_file(plumed_path):
    """
    Parses the plumed.dat file to extract KAPPA and AT values.

    Returns:
        kappa (float): The spring constant.
        at_values (list of floats): The list of loc_win_min values.
    """
    kappa = None

    with open(plumed_path, "r") as plumed_file:
        for line in plumed_file:
            # Remove comments and leading/trailing whitespace
            line = line.split("#")[0].strip()
            if "RESTRAINT" in line:
                # Extract KAPPA and AT values
                kappa_match = re.search(r"KAPPA\s*=\s*([\d\.Ee+-]+)", line)
                at_match = re.search(r"AT\s*=\s*([\d\.Ee+-]+)", line)

                if kappa_match:
                    kappa = float(kappa_match.group(1))  # No Convert to kcal/mol
                else:
                    raise ValueError("Error: KAPPA value not found in plumed.dat.")

                if at_match:
                    at_str = at_match.group(1)
                    at_value = float(at_str)
                else:
                    raise ValueError("Error: AT values not found in plumed.dat.")

                # Assuming only one RESTRAINT line, break after parsing
                break

    return kappa, at_value


def generate_metadata(
    input_dir,
    kappa,
    at_values,
    output_file="metadata.dat",
    decorrelation_time=0.02,
):
    """
    Generates a metadata file from the time series files and the plumed.dat file.

    Args:
        input_dir (str): Directory containing only input time series files (processed_COLVAR.N).
        plumed_file (str): Path to the plumed.dat file.
        output_file (str): Output metadata file name (default: 'metadata.dat').
        correlation_time (float): Placeholder value for correlation time (default: 0.02).
    """
    print(f"Generating metadata file '{input_dir}'...")
    # Validate input directory
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Error: The directory '{input_dir}' does not exist.")

    # Parse plumed.dat to get KAPPA and AT values
    logging.info(f"Parsed KAPPA: {kappa}")
    logging.info(f"Parsed {len(at_values)} AT values.")

    # Collect all processed colvar files and sort them
    files = [f for f in os.listdir(input_dir) if f.startswith("processed")]
    # Sort all the files in the files list
    print(f"Files found: {files}")
    files.sort(key=lambda x: int(x.split("_")[-1]))

    if not files:
        raise FileNotFoundError(
            f"No 'processed_COLVAR.N' files found in the directory '{input_dir}'."
        )

    if len(files) != len(at_values):
        raise ValueError(
            f"Error: Number of 'processed_COLVAR.N' files ({len(files)}) does not match number of AT values ({len(at_values)})."
        )

    # Output file path
    output_path = os.path.join(input_dir, output_file)

    # Open the metadata file for writing
    with open(output_path, "w") as metadata_file:
        for idx, filename in enumerate(files):
            filepath = os.path.join(input_dir, filename)
            loc_win_min = at_values[idx]
            spring = kappa

            # Build the metadata line with correlation time
            metadata_line = f"{filepath} {loc_win_min} {spring} {decorrelation_time}"

            # Write to the metadata file
            metadata_file.write(metadata_line + "\n")
            print(f"Added to metadata: {metadata_line}")

    print(f"\nMetadata file '{output_path}' has been generated successfully.")

    return min(at_values), max(at_values)


def plot_colvars(colvar_path="from_Archer", cutoff_time=5000, colvar_prefix="COLVAR"):
    """
    Plots the colvars data from the specified directory.

    Args:
        colvar_path (str): Path to the directory containing colvars data files (default: 'from_Archer)
    """
    print("Plotting colvars data...")
    os.chdir(colvar_path)
    # Step 1: Initialize an empty list to hold each file's CV column and the time column
    dataframes = []
    time_column = None

    # Step 2: Loop through all files that start with 'colvar' in the current directory
    for i, filename in enumerate(glob.glob(colvar_prefix + "*")):
        # Read the file, skipping the header line
        print(f"Processing file: {filename}")
        df = pd.read_csv(
            filename, sep="\s+", skiprows=1, names=["time", "CV"], on_bad_lines="warn"
        )

        df = df[df["time"] <= cutoff_time]

        # Extract the 'time' column only once, as it should be the same in all files
        if time_column is None:
            time_column = df["time"]

        # Rename the CV column to be unique for each file, e.g., 'CV_1', 'CV_2', etc.
        df = df[["CV"]].rename(columns={"CV": f"CV_{i + 1}"})
        dataframes.append(df)

    # Step 3: Combine the time column and all CV columns into a single dataframe
    combined_df = pd.concat([time_column] + dataframes, axis=1)

    # Step 4: Save the combined dataframe to a CSV file
    combined_df.to_csv("combined_colvar_data.csv", index=False)

    # Step 5: Plot the data with time on the x-axis and all CV columns on the y-axis
    plt.figure(figsize=(10, 6))
    for column in combined_df.columns[1:]:  # Skip the 'time' column
        plt.plot(combined_df["time"], combined_df[column], label=column)

    # Adding labels and legend
    plt.xlabel("Time")
    plt.ylabel("CV values")
    plt.legend()
    plt.title("CV values over time for each file")

    # Save the plot
    plt.savefig("CV_v_time.png")
    # plt.show()
    os.chdir("..")
    print("Plotting completed.")


def create_colvar_chunks(
    wham_path, colvar_100_pct_path, chunk_size_percentage=100, direction="forward"
):
    """
    Splits the colvar data files into chunks of a specified size.

    Args:
        wham_path (str): Path to the directory where the new chunk folder will be created.
        colvar_path_100pct_forward (str): Path to the directory with 100% colvar data.
        chunk_size (int): Size of each chunk (default: 100).
        direction (str): Direction of chunking ('forward' or 'reverse') (default: 'forward').

        return: chunk_path (str): Path to the directory where the chunked files are saved.
    """
    print(
        f"Creating colvar chunks for {colvar_100_pct_path} with {chunk_size_percentage}% chunk size in {direction} direction."
    )
    # Define the chunk directory and copy colvar files into it
    chunk_path = f"{wham_path}/colvars_{chunk_size_percentage}_pct_{direction}"
    if os.path.exists(chunk_path):
        logging.info(f"Chunk directory '{chunk_path}' already exists, quitting")
        return 0
    # Create chunk path
    os.makedirs(chunk_path, exist_ok=True)

    # Iterate over each file in the chunk path that starts with 'colvar'
    for filename in os.listdir(colvar_100_pct_path):
        if filename.startswith("processed_COLVAR"):
            logging.info(f"Processing file: {filename}")
            file_path = os.path.join(colvar_100_pct_path, filename)

            # Read all lines and determine chunk size
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Calculate the number of lines to keep, ensuring first line is always included
            num_lines = len(lines)
            chunk_lines = (
                int((chunk_size_percentage / 100) * (num_lines - 1)) + 1
            )  # Adjust to include header

            if direction == "forward":
                # Keep the first chunk_lines
                chunked_lines = lines[:chunk_lines]
            elif direction == "reverse":
                # Keep the last chunk_lines
                chunked_lines = [lines[0]] + lines[-(chunk_lines - 1) :]
            else:
                raise ValueError("Invalid direction. Choose 'forward' or 'reverse'.")

            out_file_path = os.path.join(chunk_path, filename)
            # Write the selected lines back to the file
            with open(out_file_path, "w") as file:
                file.writelines(chunked_lines)

    print(f"Files in '{chunk_path}' have been processed with {direction} chunking.")
    return chunk_path


def delete_processed_files(colvar_pct_folder):
    # Iterate through the files in the directory
    for filename in os.listdir(colvar_pct_folder):
        if filename.startswith(
            "processed"
        ):  # Check if the file starts with "processed"
            file_path = os.path.join(
                colvar_pct_folder, filename
            )  # Get the full file path
            try:
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


# Function to plot Coor vs Free with error as shaded region
def plot_coor_vs_free_with_shaded_error(
    coor_values, free_values, error_values, output_file="free_v3.png"
):
    """
    Plots Coor vs Free with the error as a shaded region and saves the plot to a file.

    Args:
        coor_values (numpy array): Coordinate values.
        free_values (numpy array): Free energy values.
        error_values (numpy array): Error values.
        output_file (str): Name of the output image file to save the plot (default: 'free_v3.png').
    """
    plt.figure(figsize=(10, 6))

    # Plot the line for Coor vs Free in DodgerBlue
    plt.plot(
        coor_values, free_values, label="Free Energy", color="DodgerBlue", linewidth=2
    )

    # Plot the shaded region for the error in a lighter blue
    plt.fill_between(
        coor_values,
        free_values - error_values,
        free_values + error_values,
        color="lightblue",
        alpha=0.5,
        label="Error",
    )

    # Add title and labels with larger font size
    plt.title("Coor vs Free Energy with Error (Shaded Region)", fontsize=16)
    plt.xlabel("Coor", fontsize=14)
    plt.ylabel("Free Energy", fontsize=14)

    # Add a legend and grid with subtle lines
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Set ticks to be larger and more readable
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()  # Adjust the layout for aesthetics
    plt.savefig(output_file)  # Save the plot as a PNG file


def process_free_energy_plot(file_name, output_file="free_v3.png", error_offset=0.1):
    """
    Parses the free energy data from a file and plots the free energy vs coordinate with errors.

    Args:
        file_name (str): Path to the input free energy data file.
        output_file (str): Path to save the output plot (default: 'free_v3.png').
        error_offset (float): Offset to add to the error values for visualization (default: 0.1).
    """
    coor_values, free_values, error_values = parse_free_dat_with_error(file_name)

    if len(coor_values) > 0 and len(free_values) > 0 and len(error_values) > 0:
        # Optionally modify the error values with a constant offset if needed
        plot_coor_vs_free_with_shaded_error(
            coor_values, free_values, error_values + error_offset, output_file
        )
    else:
        print("No valid data found to plot.")


def run_wham_on_path(
    hist_min,
    hist_max,
    num_bins,
    tol,
    temperature,
    numpad,
    metadata_file,
    free_file,
    num_MC_trials=1,
    rand_seed=1,
):
    """
    Runs the WHAM command with the provided arguments.

    Args:
        hist_min (float): Minimum histogram value (e.g., -1.5)
        hist_max (float): Maximum histogram value (e.g., 0)
        num_bins (int): Number of bins (e.g., 500)
        tol (float): Tolerance value (e.g., 0.5)
        temperature (float): Temperature in Kelvin (e.g., 298.15)
        numpad (int): Padding argument (e.g., 0)
        metadata_file (str): The name of the metadata file (e.g., 'metadata.dat')
        free_file (str): The name of the output free energy file (e.g., 'free_v2.dat')
        num_MC_trials (int, optional): Number of Monte Carlo trials (default: 1)
        rand_seed (int, optional): Random seed value (default: 1)
    """
    # Build the command as a list
    command = [
        "/biggin/b211/reub0138/Util/wham/wham-release-2.0.11/wham/wham/wham",
        str(hist_min),
        str(hist_max),
        str(num_bins),
        str(tol),
        str(temperature),
        str(numpad),
        metadata_file,
        free_file,
        str(num_MC_trials),
        str(rand_seed),
    ]

    # Execute the command
    try:
        result = subprocess.run(command, check=True, text=True)
        print(f"WHAM command executed successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running WHAM:\n{e.stderr}")


def process_wham_path(wham_path, at_values=None):
    """
    Processes the WHAM path by running WHAM and plotting the results.
    Args:
        wham_path (str): Path to the WHAM directory.
        plumed_path (str): Path to the plumed.dat file.
        at_values (list, optional): List of CV restraint values (default: None).
    """
    logging.info(f"Processing WHAM path: {wham_path}")

    # Run WHAM on the specified path
    run_wham_on_path(
        min(at_values) - 0.1,
        max(at_values) + 0.1,
        num_bins=500,
        tol=0.001,
        temperature=310,
        numpad=0,
        metadata_file=wham_path + "/metadata.dat",
        free_file=wham_path + "/free.dat",
        num_MC_trials=10,
        rand_seed=1,
    )

    # Parse results and plot
    coor_values, free_values, error_values = parse_free_dat_with_error(
        wham_path + "/free.dat"
    )
    plot_coor_vs_free_with_shaded_error(
        coor_values, free_values, error_values + 0.1, wham_path + "/free.png"
    )

    print("Processing completed for:", wham_path)


# Define the file parsing function
def parse_free_dat_with_error(file_name):
    """
    Parses the free energy file and extracts coordinate, free energy, and error values.

    Args:
        file_name (str): The path to the free energy file.

    Returns:
        tuple: Three numpy arrays containing the coordinate values, free energy values, and error values.
    """
    coor_values = []
    free_values = []
    error_values = []

    with open(file_name, "r") as file:
        for line in file:
            # Skip lines starting with #
            if line.startswith("#"):
                continue
            # Split line into values
            parts = line.split()
            if len(parts) == 5:  # Coor, Free, Error, Prob, Prob Error
                try:
                    coor = float(parts[0])
                    free = float(parts[1])
                    error = float(parts[2])
                    coor_values.append(coor)
                    free_values.append(free)
                    error_values.append(error)
                except ValueError:
                    # Handle any potential conversion errors
                    continue

    return np.array(coor_values), np.array(free_values), np.array(error_values)


def plot_free_combined(
    wham_path,
    structure_1="holo",
    structure_2="apo",
    units="kcal",
    box_CV_means_csv=None,
):
    """
    Plots the free energy profiles from multiple WHAM folders and saves the plot as a PNG file.
    """
    title = f"{structure_1}_{structure_2}"
    wham_paths = sorted(
        glob.glob(wham_path + "/colvars_*_pct_*"),
        key=lambda x: ("forward" in x, x),
    )
    logging.info(f"WHAM Paths: {wham_paths}")

    coor_values = []
    free_values = []
    error_values = []
    category_values = []
    data_pct_values = []

    for wham_path in wham_paths[:]:
        if not os.path.exists(wham_path + "/free.dat"):
            print(f"Skipping {wham_path} as free.dat not found.")
            continue
        coor_values_i, free_values_i, error_values_i = parse_free_dat_with_error(
            wham_path + "/free.dat"
        )
        coor_values.extend(coor_values_i)
        if units == "kcal":
            free_values_i = [i * 0.239 for i in free_values_i]
        free_values.extend(free_values_i)
        error_values.extend(error_values_i)
        foldername = wham_path.split("/")[-1]
        category_values.extend([foldername for i in range(len(coor_values_i))])
        data_pct_values.extend(
            [float(foldername.split("_")[-3]) * 0.01 for i in range(len(coor_values_i))]
        )

    print(len(coor_values))
    print(len(free_values))
    print(len(error_values))
    print(len(category_values))
    print(len(data_pct_values))
    # print(category_values)
    # Make df
    df = pd.DataFrame(
        {
            "coor": coor_values,
            "free": [i * 0.239 for i in free_values],
            "error": [i * 0.239 for i in error_values],
            "category": category_values,
            "data_pct": data_pct_values,
        }
    )

    # Define color map based on category containing 'forward' or 'reverse'
    df["colors"] = df["category"].apply(
        lambda x: "red" if "reverse" in x else "DodgerBlue"
    )

    # Get mean CV in structure_2 and structure_1 boxes
    mean_df = pd.read_csv(box_CV_means_csv)
    # Finding the structure_2 and structure_1 mean values
    structure_1_mean = mean_df[mean_df["folder"] == "sim0"]["mean"].iloc[0]
    structure_2_mean = mean_df[mean_df["folder"] == "sim23"]["mean"].iloc[0]

    # Plot coor vs free with color coding by category
    plt.figure(figsize=(8, 6), dpi=300)
    intersection_points = []  # Store intersection points data
    for category in df["category"].unique():
        category_data = df[df["category"] == category]
        plt.plot(
            category_data["coor"],
            category_data["free"],
            label=category,
            color=category_data["colors"].unique()[0],
            linewidth=2,
            alpha=category_data["data_pct"].unique()[0],
        )
        # Calculate intersections with structure_1 and structure_2 lines
        for at_value, line_label in zip(
            [structure_1_mean, structure_2_mean], [structure_1, structure_2]
        ):
            intersection = np.interp(
                at_value, category_data["coor"], category_data["free"]
            )
            intersection_points.append(
                {
                    "coor": at_value,
                    "free": intersection,
                    "category": category,
                    "line_type": line_label,
                }
            )
            plt.scatter(
                at_value, intersection, color="grey", s=20
            )  # Mark intersection on plot with specified point size
        plt.fill_between(
            category_data["coor"],
            category_data["free"] - category_data["error"],
            category_data["free"] + category_data["error"],
            color=category_data["colors"].unique()[0],
            alpha=category_data["data_pct"].unique()[0] * 0.5,
        )

    # Add a vertical line at coor = a with a label
    plt.axvline(x=structure_1_mean, color="grey", linestyle="--", linewidth=1)
    plt.axvline(x=structure_2_mean, color="grey", linestyle="--", linewidth=1)
    plt.text(
        structure_1_mean,
        plt.gca().get_ylim()[1] * 0.9,
        structure_1,
        color="grey",
        ha="right",
        fontsize=10,
        style="italic",
    )
    plt.text(
        structure_2_mean,
        plt.gca().get_ylim()[1] * 0.9,
        structure_2,
        color="grey",
        ha="right",
        fontsize=10,
        style="italic",
    )

    # Improve aesthetics
    plt.xlabel("Coor", fontsize=12)
    if units == "kcal":
        plt.ylabel("Free (kcal/mol)", fontsize=12)
    else:
        plt.ylabel("Free (kJ/mol)", fontsize=12)
    plt.title(f"STEUS for {title}", fontsize=16)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(title="Category", fontsize=10, title_fontsize="13")
    plt.tight_layout()
    if units == "kcal":
        plt.savefig(f"coor_vs_free_{title}_kcal.png")
    else:
        plt.savefig(f"coor_vs_free_{title}_kJ.png")
    plt.show()

    # Create DataFrame for intersection points and save
    intersection_df = pd.DataFrame(intersection_points)
    intersection_df.to_csv(f"intersection_points_{title}.csv", index=False)
    print(f"Intersection points saved to intersection_points_{title}.csv")
