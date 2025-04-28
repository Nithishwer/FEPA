import pandas as pd
from fepa.utils.stat_utils import calculate_metrics
from fepa.utils.abfe_utils import average_over_vanilla
from fepa.utils.plot_utils import plot_exp_v_predicted
import logging


def main():
    results_df = pd.read_csv(
        "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/exp_v_abfe_df_van_reps_all_hrex_longer_van3.csv"
    )
    avg_results_df = average_over_vanilla(
        results_df,
        keep=["Lig_Name"],
        columns_to_average=["MBAR", "BAR", "TI"],
    )
    # high_error_ligands_2 = [52542, 47821, 48951, 47594, 49599]
    # avg_results_df_high_error = avg_results_df[
    #     avg_results_df["Lig_Name"].isin(high_error_ligands_2)
    # ].copy()

    avg_results_df["overestimation"] = (
        avg_results_df["MBAR"] - avg_results_df["Experimental_G"]
    )

    # Sort by overestimation
    avg_results_df = avg_results_df.sort_values(by="overestimation", ascending=True)
    avg_results_df["Lig_Name"] = avg_results_df["Lig_Name"].astype(str)

    print(avg_results_df)

    # Plot overestimation
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Lig_Name",
        y="overestimation",
        data=avg_results_df,
        palette="viridis",
    )
    plt.xticks(rotation=90)
    plt.xlabel("Ligand Name")
    plt.ylabel("Overestimation (kcal/mol)")
    plt.title("Overestimation by Ligand Name")
    plt.tight_layout()
    plt.savefig("overestimation_by_ligand_name.png")
    plt.show()


if __name__ == "__main__":
    main()
