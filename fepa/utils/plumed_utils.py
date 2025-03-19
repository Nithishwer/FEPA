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
