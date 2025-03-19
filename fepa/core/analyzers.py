"""
This module contains a wrapper function to compute the relative entropy between two ensembles.
"""

import numpy as np
import logging
import pandas as pd
from pensa.comparison import relative_entropy_analysis


def compute_relative_entropy(
    feature_df: pd.DataFrame,
    ensemble1: str,
    ensemble2: str,
    num_bins=50,
    key="ensemble",
    feature_column_keyword=None,
):
    """
    Compute the relative entropy metrics between two ensembles.
    """
    if feature_column_keyword is not None:
        feature_names = list(
            feature_df.filter(regex=feature_column_keyword, axis=1).columns
        )
    else:
        raise ValueError("No feature column keyword provided")

    logging.info(
        "Computing Jensen-Shannon divergence for ensembles %s vs %s",
        ensemble1,
        ensemble2,
    )
    ensemble_1_df = feature_df[feature_df[key] == ensemble1]
    ensemble_1_values = ensemble_1_df.filter(
        regex=feature_column_keyword, axis=1
    ).values

    ensemble_2_df = feature_df[feature_df[key] == ensemble2]
    ensemble_2_values = ensemble_2_df.filter(
        regex=feature_column_keyword, axis=1
    ).values

    relative_entropy = relative_entropy_analysis(
        feature_names,
        feature_names,
        ensemble_1_values,
        ensemble_2_values,
        bin_num=num_bins,
    )
    return {
        "name": np.array(relative_entropy[0]),
        "jsd": np.array(relative_entropy[1]),
        "kld_ab": np.array(relative_entropy[2]),
        "kld_ba": np.array(relative_entropy[3]),
    }
