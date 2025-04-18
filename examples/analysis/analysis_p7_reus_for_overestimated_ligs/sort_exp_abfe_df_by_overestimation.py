import os
import pandas as pd

exp_abfe_csv = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/exp_v_abfe_df_van_reps_all_hrex_longer_van3.csv"

# Load the DataFrame
exp_abfe_df = pd.read_csv(exp_abfe_csv)

# Create a new overestimation column by subtracting 'MBAR' and 'Experimental_G'
exp_abfe_df["overestimation"] = exp_abfe_df["MBAR"] - exp_abfe_df["Experimental_G"]

# Sort by the overestimation column
sorted_exp_abfe_df = exp_abfe_df.sort_values(by="overestimation", ascending=True)

# Save the sorted DataFrame to a new CSV file
output_file = os.path.join("sorted_exp_abfe_df.csv")
sorted_exp_abfe_df.to_csv(output_file, index=False)
