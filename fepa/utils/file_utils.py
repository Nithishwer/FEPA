"""
This module contains utility functions for file operations.
"""

import os
import json


def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.

    Parameters:
    -----------
    config_path : str
        Path to the JSON configuration file.

    Returns:
    --------
    dict
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

def sanitize_gro_file(input_filepath: str, output_filepath: str):
    """
    Processes a .gro file to modify atom names for 'unk' residues.

    This function reads a .gro file, identifies lines corresponding to
    the 'unk' residue, and removes the 'x' character from the end of
    the atom names in those lines. The modified content is then written
    to a new output .gro file, preserving the fixed-width format.

    Args:
        input_filepath (str): The path to the input .gro file.
        output_filepath (str): The path where the modified .gro file will be saved.
    """
    try:
        with open(input_filepath, 'r') as infile:
            lines = infile.readlines()

        if len(lines) < 3:
            print(f"Warning: Input file '{input_filepath}' is too short to be a valid .gro file.")
            with open(output_filepath, 'w') as outfile:
                outfile.writelines(lines)
            return

        modified_lines = [lines[0], lines[1]] # Copy title and atom count lines

        # Process atom data lines
        for i in range(2, len(lines) - 1):
            line = lines[i]
            if len(line) >= 44: # Ensure line is long enough for coordinates
                residue_name = line[5:10].strip()
                
                if residue_name == 'unk':
                    # Extract, strip 'x', and re-pad atom name to 5 characters
                    new_atom_name = line[10:15].strip().rstrip('x').upper().rjust(5)
                    modified_lines.append(line[:10] + new_atom_name + line[15:])
                else:
                    modified_lines.append(line)
            else:
                print(f"Warning: Line {i+1} is too short to parse correctly: '{line.strip()}'")
                modified_lines.append(line)

        # Copy box dimensions line
        if len(lines) > 2:
            modified_lines.append(lines[-1])
        
        with open(output_filepath, 'w') as outfile:
            outfile.writelines(modified_lines)

        print(f"Successfully processed '{input_filepath}'. Modified content saved to '{output_filepath}'.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")