import pandas as pd
from fepa.utils.stat_utils import calculate_metrics
from fepa.utils.plot_utils import plot_exp_v_predicted
import logging


def main():
    results_df = pd.read_csv(
        "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/exp_v_abfe_df_van_reps_all_hrex_longer_van3.csv"
    )
    high_error_ligands_2 = [52542, 47821]
    results_df_high_error = results_df[
        results_df["Lig_Name"].isin(high_error_ligands_2)
    ].copy()

    metrics_dict = calculate_metrics(
        results_df_high_error["Experimental_G"], results_df_high_error["MBAR"]
    )
    logging.info(f"Metrics: {metrics_dict}")
    plot_exp_v_predicted(
        results_df_high_error,
        x="Experimental_G",
        y="MBAR",
        err_col="MBAR_Error",
        title="Deflorian MBAR",
        save_name="exp_v_predicted_mbar_higherror_top2.png",
        metrics_dict=metrics_dict,
        xlim=(-20, 0),
        ylim=(-20, 0),
        color_by="Lig_Name",
        label=True,
    )
    plot_exp_v_predicted(
        results_df_high_error,
        x="Experimental_G",
        y="MBAR",
        err_col="MBAR_Error",
        title="Deflorian MBAR",
        save_name="exp_v_predicted_mbar_higherror_top2_vanilla.png",
        metrics_dict=metrics_dict,
        xlim=(-20, 0),
        ylim=(-20, 0),
        color_by="Vanilla",
        label=True,
    )

    high_error_ligands_3 = [48951, 47594, 49599]
    results_df_high_error = results_df[
        results_df["Lig_Name"].isin(high_error_ligands_3)
    ].copy()

    metrics_dict = calculate_metrics(
        results_df_high_error["Experimental_G"], results_df_high_error["MBAR"]
    )
    logging.info(f"Metrics: {metrics_dict}")
    plot_exp_v_predicted(
        results_df_high_error,
        x="Experimental_G",
        y="MBAR",
        err_col="MBAR_Error",
        title="Deflorian MBAR",
        save_name="exp_v_predicted_mbar_higherror_next3.png",
        metrics_dict=metrics_dict,
        xlim=(-20, 0),
        ylim=(-20, 0),
        color_by="Vanilla",
        label=True,
    )

    selection = results_df["Lig_Name"].isin(high_error_ligands_2 + high_error_ligands_3)
    results_df_others = results_df[[not i for i in selection]].copy()

    metrics_dict = calculate_metrics(
        results_df_others["Experimental_G"], results_df_others["MBAR"]
    )
    logging.info(f"Metrics: {metrics_dict}")
    plot_exp_v_predicted(
        results_df_others,
        x="Experimental_G",
        y="MBAR",
        err_col="MBAR_Error",
        title="Deflorian MBAR",
        save_name="exp_v_predicted_mbar_others.png",
        metrics_dict=metrics_dict,
        xlim=(-20, 0),
        ylim=(-20, 0),
        color_by="Vanilla",
        label=False,
    )


if __name__ == "__main__":
    main()
