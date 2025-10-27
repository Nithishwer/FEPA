import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path  

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_paths_for_compound, load_abfe_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.featurizers import BPWaterFeaturizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# === NEW: helper to absolutize path templates from the test config ===
def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative templates in the test config."""
    keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]
    out = dict(config)
    for k in keys:
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out
# === END helper ===


def main():
    """Main function to run the analysis on truncated test data."""

    # === MOD: locate FEPA repo root ===
    repo_root = Path(__file__).resolve().parents[3]

    # === MOD: load the test config from FEPA/tests/test_config/config.json ===
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    config = load_config(str(cfg_path))
    config = _abspath_templates(config, repo_root)

    # === MOD: set output directory to FEPA/tests/test_data/2_expected ===
    analysis_output_dir = (repo_root / "tests" / "test_data" / "2_expected").resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    # === END MOD BLOCK ===

    # Loop over truncated compound list
    for cmp in config["compounds"][:2]:
        logging.info("Analyzing compound %s ...", cmp)

        # Create compound subdirectory
        cmp_output_dir = analysis_output_dir / cmp
        os.makedirs(cmp_output_dir, exist_ok=True)

        # Prepare paths (unchanged)
        logging.info("Loading paths for compound %s...", cmp)
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=[1],
            leg_window_list = [f"coul.{i:02}" for i in range(2)],
            # [f"rest.{i:02d}" for i in range(0, 12)]
            # [f"vdw.{i:02d}" for i in range(0, 21)],
            bp_selection_string=(
                "name CA and resid 54 55 56 57 58 59 60 61 62 64 65 68 83 84 85 87 88 91 92 173 176 177 180 217 218 221 225 235 238 239 240 241 242 243 244 245 246 247"
            ),
            apo=False,
        )

        print(path_dict)

        # Load trajectories
        logging.info("Loading trajectories for compound %s ...", cmp)
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()

        # Featurize
        logging.info("Featurizing binding pocket waters ...")
        bp_water_featurizer = BPWaterFeaturizer(ensemble_handler=ensemble_handler)
        bp_water_featurizer.featurize(radius=10)

        # Save features
        logging.info("Saving features for compound %s ...", cmp)
        bp_water_featurizer.save_features(cmp_output_dir, overwrite=True)

        # Read features
        features_df = pd.read_csv(cmp_output_dir / "WaterOccupancy_features.csv")

        # Plot per-vanilla time series
        for van in [1]:
            van_features_df = features_df[
                features_df["ensemble"].str.contains(f"van_{van}")
            ]
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=van_features_df, x="Time (ps)", y="occupancy", hue="ensemble"
            )
            plt.title(f"Water Occupancy for {cmp}")
            plt.xlabel("Time (ps)")
            plt.xlim(0, 20000)
            plt.ylabel("Number of Waters")
            plt.legend(
                title="Ensemble",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                ncol=2,
            )
            plt.tight_layout()
            plt.savefig(cmp_output_dir / f"{cmp}_water_occupancy_van{van}_timeseries_v2.png")

        # Group by ensemble and calculate average occupancy
        features_df["van"] = features_df["ensemble"].str.extract(r"van_(\d)")
        features_df["id"] = features_df["ensemble"].str.replace(
            r"_van_\d+", "", regex=True
        )
        avg_df = features_df.groupby(["id", "van"], as_index=False)["occupancy"].mean()

        # Plot average across windows
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=avg_df,
            x="id",
            y="occupancy",
            hue="van",
            palette="tab10",
        )
        plt.title(
            f"Average Water Occupancy Across Windows for {cmp}",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Window ID", fontsize=14)
        plt.ylabel("Average Number of Waters", fontsize=14)
        plt.legend(
            title="Vanilla Repeat",
            title_fontsize=12,
            fontsize=10,
            loc="upper right",
        )
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(
            cmp_output_dir / f"{cmp}_water_occupancy_across_windows_v2.png",
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    main()
