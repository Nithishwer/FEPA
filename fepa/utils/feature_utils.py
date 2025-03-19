"""
Utility functions for feature dataframes.
"""

import re
from fepa.core.analyzers import compute_relative_entropy

# def get_only_feature_df_values(feature_df, feature_column_keyword=None):
#     """
#     Returns the values of the feature dataframe with only feature columns.
#     """
#     only_feature_df = feature_df.filter(regex=feature_column_keyword, axis=1)
#     return only_feature_df.values


# def get_feature_df_subset_values(feature_df, key, value, feature_column_keyword=None):
#     """
#     Returns a subset of the feature dataframe based on a key-value pair.
#     """
#     subset_feature_df = feature_df[feature_df[key] == value]
#     if feature_column_keyword is not None:
#         subset_feature_df_values = subset_feature_df.filter(
#             regex=feature_column_keyword, axis=1
#         ).values
#     else:
#         subset_feature_df_values = subset_feature_df.drop(
#             columns=["timestep", "ensemble"]
#         ).values
#     return subset_feature_df_values


def get_mda_selection_string_from_sdf_names(sdf_names):
    """
    Extracts the residue name and ID from a self distance featurizer name containing both.
    """
    pattern = r"DIST: (\w+) (\d+) (\w+) - (\w+) (\d+) (\w+)"
    selection_list = []
    for sdf_name in sdf_names:
        match = re.match(pattern, sdf_name)
        if not match:
            raise ValueError("Invalid distance string format")
        resname1, resid1, atom1, resname2, resid2, atom2 = match.groups()
        selection1 = f"resname {resname1} and resid {resid1} and name {atom1}"
        selection2 = f"resname {resname2} and resid {resid2} and name {atom2}"
        selection_list.append((selection1, selection2))

    return selection_list


def get_resid_pairs_from_sdf_names(sdf_names):
    """
    Extracts the resid pairs from a self distance featurizer name containing both.
    """
    pattern = r"DIST: (\w+) (\d+) (\w+) - (\w+) (\d+) (\w+)"
    resid_pairs = []
    for sdf_name in sdf_names:
        match = re.match(pattern, sdf_name)
        if not match:
            raise ValueError("Invalid distance string format")
        resname1, resid1, atom1, resname2, resid2, atom2 = match.groups()
        resid_pairs.append((int(resid1), int(resid2)))

    return resid_pairs


# def extract_resno_from_dist_names(distance_names):
#     """
#     Extracts the residue number from a list of distance feature names.
#     """
#     # Regular expression pattern to extract residue numbers
#     pattern = r"\b[A-Z]{3} (\d+) CA\b"

#     # Extract residue numbers as tuples
#     residue_pairs = [
#         tuple(map(int, re.findall(pattern, distance_name)))
#         for distance_name in distance_names
#     ]

#     return residue_pairs


def filter_top_features(
    feature_df, key, ensemble1, ensemble2, feature_column_keyword, top_n=10
):
    jsd_dict = compute_relative_entropy(
        feature_df=feature_df,
        ensemble1=ensemble1,
        ensemble2=ensemble2,
        key=key,
        num_bins=50,
        feature_column_keyword=feature_column_keyword,  # To make sure everything except the column names with the keyword is deleted when making the pca object
    )
    # Select top features according to JS entropy
    top_indices = jsd_dict["jsd"].argsort()[-top_n:][::-1]
    top_features_names = jsd_dict["name"][top_indices]
    # KEep only the clolumns with numbers in top indices in feature df
    top_features_df = feature_df[top_features_names]
    top_features_df["cluster"] = feature_df["cluster"]
    top_features_df["ensemble"] = feature_df["ensemble"]
    top_features_df["timestep"] = feature_df["timestep"]
    return top_features_df
