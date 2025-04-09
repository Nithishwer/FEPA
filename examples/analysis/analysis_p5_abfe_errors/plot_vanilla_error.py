import pandas as pd
import logging
import matplotlib.pyplot as plt


def main():
    results_df = pd.read_csv(
        "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/exp_v_abfe_df_van_reps_all_hrex_longer_van3.csv"
    )
    # Convert the "Lig_Name" column to string type
    # This is important to avoid any issues with the unique() function
    # and to ensure that the column is treated as a categorical variable
    # in the DataFrame
    results_df["Lig_Name"] = results_df["Lig_Name"].astype(str)
    unique_ligands = results_df["Lig_Name"].unique()
    ligand_error_dict = {"lig_name": [], "Error": []}

    for ligand in unique_ligands:
        ligand_df = results_df[results_df["Lig_Name"] == ligand]
        ligand_error_dict["lig_name"].append(ligand)
        ligand_error_dict["Error"].append(
            max(ligand_df["MBAR"]) - min(ligand_df["MBAR"])
        )

    ligand_error_df = pd.DataFrame(ligand_error_dict)

    # sort and plot the errors as bar chart
    ligand_error_df = ligand_error_df.sort_values(by="Error", ascending=False)
    print(ligand_error_dict)
    plt.figure(figsize=(12, 6))  # Adjust the figure size to prevent overlap
    ligand_error_df.plot(
        x="lig_name",
        y="Error",
        kind="bar",
        title="Ligand Errors",
        xlabel="lig_name",
        ylabel="Error",
    )
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels and align them
    plt.tight_layout()  # Automatically adjust subplot parameters to fit
    plt.show()
    # Save the plot
    plt.savefig("ligand_errors.png")
    # Save the dataframe
    ligand_error_df.to_csv("ligand_errors.csv", index=False)


if __name__ == "__main__":
    main()
