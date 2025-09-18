import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis import Merge
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.featurizers import BPWaterFeaturizer
from MDAnalysis.coordinates.memory import MemoryReader
import MDAnalysis.transformations as trans

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the analysis"""

    # Load configuration
    config_path = os.path.join("../../config/config.json")
    config = load_config(config_path)
    analysis_output_dir = os.path.join("a3a_compare_bp_water_ligand_overlap", "wdir")
    overlap_cutoff = 2.0  # in angstroms

    # for cmp in config["compounds"][:1]:
    for cmp in ["9"]:
        # Log
        logging.info("Analyzing compound %s ...", cmp)

        # Create output directory
        cmp_output_dir = os.path.join(analysis_output_dir, cmp)
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths
        logging.info("Loading paths for compound %s...", cmp)
        path_dict = load_paths_for_compound(
            config,
            cmp,
            bp_selection_string="name CA and resid 57 58 61 64 83 84 87 88 91 92 173 177 218 221 235 238 239 242 243 246",  # Using V740 and S809 COM adjusted for start (-567) to define the binding pocket
            apo=True,
        )

        print(path_dict)

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()

        # Align holo universe with apo universe
        logging.info("Aligning holo universe with apo universe ...")
        holo_universe = ensemble_handler.universe_dict[f"{cmp}_van_1"]
        # holo_universe = mda.Universe(
        #     "/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/mGlu5_9/vanilla_rep_1/npt.gro",
        #     #    "/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/mGlu5_9/vanilla_rep_1/prod.xtc",
        # )
        holo_universe.atoms.write(
            os.path.join(cmp_output_dir, f"{cmp}_holo_universe.pdb")
        )

        # Last frame of the holo universe
        holo_universe.trajectory[-1]
        holo_frame_universe = mda.Merge(holo_universe.atoms)
        complex_selection = holo_frame_universe.select_atoms("protein or resname unk")
        complex_selection.write(os.path.join(cmp_output_dir, f"holo_{cmp}_complex.pdb"))

        # Overlapping waters dict
        overlapping_waters_dict = {}

        for apo in [1, 2, 3]:
            apo_mobile_universe = ensemble_handler.universe_dict[f"apo_{apo}"]
            # apo_mobile_universe = mda.Universe(
            #     f"/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/apo_mGlu5_r1/topol.tpr",
            #     f"/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/apo_mGlu5_r1/prod_short.xtc",
            # )
            # PBC correction
            # protein = apo_mobile_universe.select_atoms("protein")
            # not_protein = apo_mobile_universe.select_atoms("not protein")
            # transforms = [
            #     trans.unwrap(protein),
            #     trans.center_in_box(protein, wrap=True),
            #     trans.unwrap(not_protein),
            # ]
            # apo_mobile_universe.trajectory.add_transformations(*transforms)

            # RMSD beofre aligning
            unaligned_rmsd = rms.rmsd(
                apo_mobile_universe.select_atoms("protein").positions,
                holo_frame_universe.select_atoms("protein").positions,
                superposition=False,
            )
            print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")

            # Align the apo trajectory to the holo trajectory
            alignment = align.AlignTraj(
                apo_mobile_universe,
                holo_frame_universe,
                select="protein and name CA",
                filename=f"rmsfit_{apo}.dcd",
                # in_memory=True,
            )

            logging.info("Aligning apo_%d trajectory to holo trajectory ...", apo)
            alignment.run()

            # Load the aligned trajectory
            rms_fit_apo_universe = mda.Universe(
                #                path_dict[f"apo_{apo}"]["tpr"],
                path_dict[f"apo_{apo}"]["tpr"],
                f"rmsfit_{apo}.dcd",
            )

            # Calculate RMSD after aligning
            aligned_rmsd = rms.rmsd(
                rms_fit_apo_universe.select_atoms("protein").positions,
                holo_frame_universe.select_atoms("protein").positions,
                superposition=True,
            )
            print(f"Aligned RMSD: {aligned_rmsd:.2f}")
            # Select only the ligand from holo
            ligand_selection = holo_frame_universe.select_atoms("resname unk")
            all_coords = []

            # Iterate through both trajectories in parallel
            for ts1 in rms_fit_apo_universe.trajectory:
                # Copy current positions of ligand
                ligand_coords = ligand_selection.positions.copy()
                # Concatenate ligand coords to u2 positions
                combined_coords = np.vstack((ts1.positions, ligand_coords))
                all_coords.append(combined_coords)

            # Merge u2 atoms and u1 ligand atoms
            combined = Merge(rms_fit_apo_universe.atoms, ligand_selection)

            # Assign the collected coordinates to the new Universe
            combined.load_new(all_coords, order="fac")

            # Overlapping waters array
            Overlapping_waters_array = []
            time_array = []

            # log the number length of each trajectory
            logging.info(
                f"Total time of combined {apo} trajectory: {combined.trajectory[-1].time}"
            )
            logging.info(
                f"Total number of frames in combined {apo} trajectory: {len(combined.trajectory)}"
            )
            timestep = 0
            for ts in combined.trajectory:
                # Increment the timestep
                timestep += rms_fit_apo_universe.trajectory.dt
                # Count the no of water molecules within 2A of any ligand atom
                overlapping_waters = combined.select_atoms(
                    f"resname SOL and around {overlap_cutoff} (resname unk)"
                ).residues.atoms
                # extend selection to complete molecules

                # Count the number of overlapping waters
                num_overlapping_waters = len(overlapping_waters) / 3
                logging.info(
                    f"Number of overlapping waters within {overlap_cutoff} angstroms of the ligand in apo %d at time %f: %d",
                    apo,
                    timestep,
                    num_overlapping_waters,
                )
                # Append to the array
                Overlapping_waters_array.append(num_overlapping_waters)
                time_array.append(timestep)

                # if there are more than 4 waters, save the coordinates as pdb
                if num_overlapping_waters > 4:
                    # if timestep == 800:
                    rms_fit_apo_universe.trajectory[ts.frame]
                    logging.info(
                        f"Saving overlapping waters for apo {apo} at time {timestep} with {num_overlapping_waters} waters"
                    )
                    rms_fit_apo_universe.select_atoms("protein").write(
                        os.path.join(
                            cmp_output_dir, f"apo_{apo}_frame_{timestep}_protein.gro"
                        )
                    )
                    combined.select_atoms(
                        f"resname SOL and around {overlap_cutoff} (resname unk)"
                    ).write(
                        os.path.join(
                            cmp_output_dir,
                            f"apo_{apo}_frame_{timestep}_waters.gro",
                        )
                    )

            # if there is no time array in overlapping_Waters_dict, create one
            if "time" not in overlapping_waters_dict:
                overlapping_waters_dict["time"] = [i / 1000 for i in time_array]
            # Add the overlapping waters array to the dict
            overlapping_waters_dict[f"apo_{apo}"] = Overlapping_waters_array

        # Convert the dict to a dataframe
        overlapping_waters_df = pd.DataFrame(overlapping_waters_dict)
        overlapping_waters_df.set_index("time", inplace=True)

        # Plotting
        overlapping_waters_df.plot(
            figsize=(10, 6)
        )  # Automatically assigns different colors
        plt.xlabel("Time (ns)")
        plt.ylabel("Value")
        plt.title(
            f"Water within {overlap_cutoff} A of ligand positions in apo simulations over Time"
        )
        plt.legend(title="Apo Columns")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(cmp_output_dir, f"{cmp}_apo_waters_overlap_timeseries.png")
        )
        plt.close()


if __name__ == "__main__":
    main()
