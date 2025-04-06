import os
import shutil
import fepa
import logging
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from matplotlib import cm
from fepa.utils.path_utils import load_paths_for_memento_equil
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SelfDistanceFeaturizer
from fepa.utils.md_utils import check_bp_residue_consistency

logging.basicConfig(level=logging.INFO)


def plot_colvar_histogram(colvar_df, save_path):
    """
    Plots a histogram of the collective variables from the dataframe and saves it to the specified path.
    """

    colvar_df.plot.hist(
        alpha=0.5, bins=100, edgecolor="black", figsize=(10, 6), colormap="tab10"
    )
    plt.title("Histogram of Collective Variables")
    plt.xlabel("CV Value")
    plt.ylabel("Frequency")
    # Sort legend labels by the number after 'sim'
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_labels_handles = sorted(
        zip(labels, handles), key=lambda x: int(x[0].replace("sim", ""))
    )
    sorted_labels, sorted_handles = zip(*sorted_labels_handles)
    plt.legend(
        sorted_handles,
        sorted_labels,
        title="Simulations",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=2,  # Set number of columns in the legend
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def run_plumed_on_traj(plumed_file, xtc_file):
    """
    Extracts the CV value through the simulation with the plumed file.
    """
    # Run the plumed driver on the simulation
    subprocess.run(
        ["plumed", "driver", "--plumed", plumed_file, "--mf_xtc", xtc_file], check=True
    )
    logging.info(f"Completed plumed driver run on {xtc_file}.")


def read_cv_column(filename):
    df = pd.read_csv(filename, delim_whitespace=True, comment="#", names=["time", "CV"])
    return df["CV"].tolist()


def analyse_memento_equil_w_plumed(memento_dir, reference_pdb):
    # Navigate to memento_dir/wdir/boxes
    print(f"Setting up equilibration for the complex in {memento_dir}.")
    boxes_dir = os.path.join(memento_dir, "wdir", "boxes")
    plumed_path = os.path.join(boxes_dir, "plumed.dat")

    # Copy the box files to the memento_dir
    shutil.copy(reference_pdb, boxes_dir)

    # Create a colvar dict
    colvar_dict = {}

    # Iterate through folders starting with 'sim'
    for folder_name in os.listdir(boxes_dir):
        if folder_name.startswith("sim"):
            sim_path = os.path.join(boxes_dir, folder_name)
            current_dir = os.getcwd()
            os.chdir(sim_path)

            # Run the plumed driver on the simulation
            run_plumed_on_traj(plumed_path, "prod.xtc")

            # load CV from COLVAR file
            colvar_file = os.path.join(sim_path, "COLVAR")
            colvar_dict[folder_name] = read_cv_column(colvar_file)

            # Return to the original directory
            os.chdir(current_dir)

    # Convert dict into a dataframe and plot
    colvar_df = pd.DataFrame(colvar_dict)

    # Plot plumed CV histogram
    plot_colvar_histogram(colvar_df, os.path.join(boxes_dir, "colvar_histogram.png"))
    logging.info(f"Plotted histogram from plumed for {memento_dir}.")


def analyse_memento_equil_w_fepa(
    memento_dir, bp_selection_string, pca_pickle_path, top_dist_names=None
):
    # Navigate to memento_dir/wdir/boxes
    boxes_dir = os.path.join(memento_dir, "wdir", "boxes")
    sim_path_template = os.path.join(boxes_dir, "{SIM_FOLDER}")
    path_dict = load_paths_for_memento_equil(sim_path_template, bp_selection_string)
    logging.info(f"Loaded paths for memento equilibration: {memento_dir}.")
    # Load trajectories
    ensemble_handler = EnsembleHandler(path_dict=path_dict)
    ensemble_handler.make_universes()
    check_bp_residue_consistency(ensemble_handler.get_universe_dict())
    logging.info(f"Loaded trajectories for memento equilibration: {memento_dir}.")
    # Featurize all
    featurizer = SelfDistanceFeaturizer(ensemble_handler)
    featurizer.featurize()
    logging.info(f"Featurized trajectories for memento equilibration: {memento_dir}.")
    # Get top feature df
    if top_dist_names is not None:
        feature_df = featurizer.get_feature_df()
        top_feature_df = feature_df[top_dist_names]
        dimreducer = PCADimReducer(top_feature_df, n_components=8)
    else:
        dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
    dimreducer.load_pca(save_path=pca_pickle_path)
    dimreducer.calculate_projections()
    dimreducer.save_projection_df(
        save_path=os.path.join(boxes_dir, f"pca_projection_df.csv")
    )
    pc_df = dimreducer.get_pca_projection_df()
    pc_df["ensemble"] = pc_df["ensemble"].astype(str)

    # Plot PC1 Histogram
    ax = sns.histplot(
        pc_df, x="PC1", hue="ensemble", bins=100, legend=True, palette="tab10"
    )
    # Set legend correctly
    # plt.legend(title="Ensemble", loc="upper left")

    # Customize plot
    plt.xlabel("PC1")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.title("Histogram of PC1 by Ensemble")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=2, title_fontsize=14)

    # Show plot
    plt.savefig(
        os.path.join(boxes_dir, "colvar_histogram_fepa.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    logging.info(f"Plotted PC1 histogram for {memento_dir}.")


def main():
    # Example: You can replace the input path with a dynamic input mechanism if needed.
    apo_pairs = [("apo_1", "apo_2"), ("apo_1", "apo_3"), ("apo_2", "apo_3")]
    reference_pdb = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/fepa/examples/wdir/data/analysis_p4_reduce_all_apo/reference.pdb"
    bp_selection_string = "name CA and resid 12  54  57  58  59  60  61  62  64  65  66  71  77  78  81  82  83  84  85  86  89  90  135  138  141  142  161  162  163  174  175  178  182  232  235  236  238  239  242  254  261  264  265  266  268  269"
    memento_run_name = "memento_run_v1"
    analysis_dir = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/fepa/examples/wdir/data/analysis_p4_reduce_all_apo"
    pickle_template = analysis_dir + "/{PAIR}/pca_top_{PAIR}.pkl"
    top_features_csv = analysis_dir + "/{PAIR}/top_features_{PAIR}_sample.csv"
    for apo_pair in apo_pairs[:1]:
        logging.info(f"Analyzing the apo pair: {apo_pair}.")
        memento_dir = f"/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1_memento/{apo_pair[0]}_{apo_pair[1]}/{memento_run_name}"
        analyse_memento_equil_w_plumed(memento_dir, reference_pdb)
        col_names = pd.read_csv(
            top_features_csv.format(PAIR=f"{apo_pair[0]}_{apo_pair[1]}")
        ).columns.tolist()
        col_names_dist = [col for col in col_names if "DIST" in col] + [
            "ensemble",
            "timestep",
        ]
        analyse_memento_equil_w_fepa(
            memento_dir,
            bp_selection_string,
            pca_pickle_path=pickle_template.format(PAIR=f"{apo_pair[0]}_{apo_pair[1]}"),
            top_dist_names=col_names_dist,
        )


# Ensure the script runs only when executed as the main program
if __name__ == "__main__":
    main()
