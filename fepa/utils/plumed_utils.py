import re
import numpy as np
import logging
from fepa.utils.feature_utils import get_resid_pairs_from_sdf_names


def write_plumed_file(sdf_names, top_features_pca, save_path, molinfo_structure):
    # Get residue pairs from the top feature df columns
    resid_pairs = get_resid_pairs_from_sdf_names(sdf_names)

    # Plumed text
    plumed_text = f"MOLINFO STRUCTURE={molinfo_structure}\n"

    # Add distances to the plumed_text
    for i, resid_pair in enumerate(resid_pairs, start=1):
        res1, res2 = resid_pair
        line = f"d{i}: DISTANCE ATOMS=@CA-{res1},@CA-{res2}\n"
        plumed_text += line

    # Dot product with dynamic number of terms based on top_n
    pc1 = top_features_pca.components_[0]

    combine_args = ",".join([f"d{i}" for i in range(1, len(pc1) + 1)])
    coefficients = ",".join(map(str, pc1))
    dot_product_text = f"""# Create the dot product
dot: COMBINE ARG={combine_args} COEFFICIENTS={coefficients} PERIODIC=NO
CV: MATHEVAL ARG=dot FUNC=10*x PERIODIC=NO"""
    # print(dot_product_text)
    plumed_text += dot_product_text + "\n"

    # Save the output in colvar
    save_text = "PRINT ARG=CV FILE=COLVAR STRIDE=1"
    plumed_text += save_text + "\n"

    # Restraints
    restraint_text = """# Put position of restraints here for each window
restraint: RESTRAINT ARG=CV AT=@replicas:$RESTRAINT_ARRAY KAPPA=$KAPPA
PRINT ARG=restraint.* FILE=restr
"""
    plumed_text += restraint_text + "\n"

    # Save the plumed file
    if save_path:
        with open(save_path, "w") as f:
            f.write(plumed_text)
        print(f"Plumed file saved at {save_path}")


def write_plumed_restraints(plumed_file, restraint_centers, kappa):
    # This function reads a plumed file and replaces the $RESTRAINT_ARRAY and $KAPPA with the restraint_centers and kappa
    with open(plumed_file, "r") as f:
        plumed_text = f.read()

    restraint_centers_str = ",".join(map(str, restraint_centers))
    new_plumed_text = re.sub(r"\$RESTRAINT_ARRAY", restraint_centers_str, plumed_text)
    new_plumed_text = re.sub(r"\$KAPPA", str(kappa), new_plumed_text)

    if new_plumed_text == plumed_text:
        logging.warning("No replacements were made in the plumed file.")
    else:
        with open(plumed_file, "w") as f:
            f.write(new_plumed_text)
        logging.info(f"Plumed restraints written to {plumed_file}")

        def make_restraint_array(
            ensemble_centers_df, key, ensemble1, ensemble2, kappa, CV_column="PC1"
        ):
            """
            Make restraint array for plumed file
            """
            CV_value_ensemble1 = ensemble_centers_df[CV_column][
                ensemble_centers_df[key] == ensemble1
            ].values[0]
            CV_value_ensemble2 = ensemble_centers_df[CV_column][
                ensemble_centers_df[key] == ensemble2
            ].values[0]

            # Generate a equidistant array from CV_value_ensemble1 to CV_value_ensemble2
            restraint_centers = np.linspace(
                CV_value_ensemble1, CV_value_ensemble2, num=24
            )

            return restraint_centers


def make_restraint_array_from_ensemble_centers(
    ensemble_centers_df, key, ensemble1, ensemble2, kappa, CV_column="PC1"
):
    """
    Make restraint array for plumed file
    """
    CV_value_ensemble1 = ensemble_centers_df[CV_column][
        ensemble_centers_df[key] == ensemble1
    ].values[0]
    CV_value_ensemble2 = ensemble_centers_df[CV_column][
        ensemble_centers_df[key] == ensemble2
    ].values[0]
    # Generate a equidistant array from CV_value_ensemble1 to CV_value_ensemble2
    restraint_centers = np.linspace(CV_value_ensemble1, CV_value_ensemble2, num=24)
    return restraint_centers


def add_resid_offset_to_ca_indices(input_path, output_path, offset, resid_break):
    # Pattern to match @CA-XXX where XXX is one or more digits
    pattern = re.compile(r"@CA-(\d+)")

    with open(input_path, "r") as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        # Replace each @CA-XXX with @CA-(XXX + offset) or @CA-(XXX + offset + resid_break[1]) based on condition
        def replace_ca(match):
            if resid_break is None:
                first_missing_resid = 0
                no_of_missing_residues = 0
            else:
                first_missing_resid = resid_break[0]
                no_of_missing_residues = resid_break[1]
            # Convert the resids
            resid = int(match.group(1))
            if resid < first_missing_resid - offset:
                new_resid = resid + offset
            else:
                new_resid = resid + offset + no_of_missing_residues
            return f"@CA-{new_resid}"

        new_line = pattern.sub(replace_ca, line)
        modified_lines.append(new_line)

    with open(output_path, "w") as file:
        file.writelines(modified_lines)


def remove_duplicate_headers_and_clean(input_path, output_path):
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    with open(output_path, "w") as outfile:
        for i, line in enumerate(lines):
            if i == 0 or (not line.startswith("#") and len(line.split(" ")) == 2):
                outfile.write(line)
