import os
import MDAnalysis as mda
from fepa.utils.memento_utils import run_pymemento, prepare_memento_input


def main():
    # Declaring variables:
    memento_dir = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1_memento"

    # Defining pairs
    apo_pairs = [("apo_1", "apo_2"), ("apo_1", "apo_3"), ("apo_2", "apo_3")]
    inital_target_gro_dict = {
        "apo_1_apo_2": ("cluster_0_center.gro", "cluster_1_center.gro"),
        "apo_1_apo_3": ("cluster_0_center.gro", "cluster_1_center.gro"),
        "apo_2_apo_3": ("cluster_0_center.gro", "cluster_1_center.gro"),
    }
    analysis_output_dir = "/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/fepa/examples/wdir/data/analysis_p4_reduce_all_apo"

    # Loop through the list and create folders
    for pair in apo_pairs:
        folder_name = f"{pair[0]}_{pair[1]}"
        folder_path = os.path.join(memento_dir, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_name}' created successfully.")
        else:
            print(f"Folder '{folder_name}' already exists.")
            os.chdir(folder_path)

        # Defining initial and final gro paths
        initial_gro = os.path.join(
            analysis_output_dir, folder_name, inital_target_gro_dict[folder_name][0]
        )
        target_gro = os.path.join(
            analysis_output_dir, folder_name, inital_target_gro_dict[folder_name][1]
        )

        prepare_memento_input(
            initial_gro=initial_gro, target_gro=target_gro, run_name="memento_run_v1"
        )

        os.chdir("memento_run_v1")
        run_pymemento(
            template_path="/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1_memento/apo_template",
            last_run="processing",
            protonation_states=[1, 1, 1, 2, 1, 2],
        )
        os.chdir("../..")


if __name__ == "__main__":
    main()
