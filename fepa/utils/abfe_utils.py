import pandas as pd
import numpy as np


def average_over_vanilla(
    df, keep=["Lig_Name"], columns_to_average=["MBAR", "BAR", "TI"]
):
    """
    Make a DataFrame by averaging over vanilla and reps,
    and add averaging error columns for specified columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        keep (list): Columns to group by.
        columns_to_average (list): Columns to calculate averaging errors for.

    Returns:
        pd.DataFrame: DataFrame with averaged values and averaging error columns.
    """
    # Compute averaging errors (max-min)/2 for the specified columns
    for col in columns_to_average:
        print("Averaging column:", col)
        error_col = f"{col}_averaging_error"
        df[error_col] = df.groupby(keep)[col].transform(
            lambda x: (x.max() - x.min()) / 2
        )

    # Group by the specified columns and calculate the mean values for each group
    df = df.drop(columns=["Rep", "Vanilla"])
    grouped_df = df.groupby(keep)
    avg_df = grouped_df
    avg_df = grouped_df.agg(
        {
            "Experimental_G": "first",  # Keep 'Experimental_G' as it's constant for each group
            "MBAR": "mean",
            "MBAR_Error": "mean",
            "BAR": "mean",
            "BAR_Error": "mean",
            "TI": "mean",
            "TI_Error": "mean",
            "MBAR_averaging_error": "first",
            "BAR_averaging_error": "first",
            "TI_averaging_error": "first",
            "Convergence": "all",
        }
    ).reset_index()

    return avg_df
