# tests/test_fipa_final_integration.py
"""
Final FEPA integration test (semantic reproduction of the provided script).

What this test does:
- Reads existing, read-only inputs exactly where the script expects them:
  * FEPA config at /biggin/b230/magd5710/Documents/FEPA/tests/test_config/config.json
  * Cluster assignment CSVs under tests/test_data/6_expected/<cmp>/<cmp>_conformation_cluster_df.csv
  * External exp vs ABFE CSV at /biggin/b211/reub0138/Projects/orexin/christopher_et_al_v2_solvated/exp_v_abfe_df_van_reps_all_hrex.csv
- Performs ONLY the analysis & plotting logic of the script (no module import/exec).
- Writes ALL outputs (combined CSV and plot) ONLY into pytest's tmp_path.
- Asserts output files exist (existence checks only).

We avoid line-by-line reproduction; instead we call FEPA public APIs and standard
libraries within fixtures/tests to match the script's semantics.
"""
from __future__ import annotations

# ---- Guards: disable bytecode, silence warnings, force non-GUI matplotlib BEFORE other imports ----
import sys as _sys, os as _os, warnings as _warnings
_sys.dont_write_bytecode = True
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_os.environ["MPLBACKEND"] = "Agg"
_warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# FEPA public APIs (read-only usage)
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound  # used to mirror path resolution (no writes)

# Logger (explicit logging required)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------- Fixtures (read-only inputs) -----------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute FEPA repository root (read-only inputs live here)."""
    root = Path("/biggin/b230/magd5710/Documents/FEPA").resolve()
    logger.info("Using FEPA repo root: %s", root)
    return root


@pytest.fixture(scope="session")
def config_and_inputs(repo_root: Path) -> Tuple[dict, List[str], Path, Path]:
    """
    Load FEPA test config, absolutize templates in-memory, and return:
      (config, compounds, expected_root, exp_vs_abfe_csv_path)
    """
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    cfg = load_config(str(cfg_path))

    # Absolutize relative templates in-memory (no writes to disk)
    for k in ("abfe_window_path_template", "vanilla_path_template", "vanilla_path_template_old", "apo_path_template"):
        if k in cfg:
            p = Path(cfg[k])
            if not p.is_absolute():
                cfg[k] = str((repo_root / p).resolve())

    compounds = sorted(list(cfg["compounds"]))
    if not compounds:
        raise RuntimeError("No compounds found in FEPA test config.")

    expected_root = repo_root / "tests" / "test_data" / "6_expected"
    exp_vs_abfe_csv = Path(
        "/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v2_solvated/exp_v_abfe_df_van_reps_all_hrex.csv"
    ).resolve()

    logger.info("Resolved expected_root=%s; exp_vs_abfe_csv=%s", expected_root, exp_vs_abfe_csv)
    return cfg, compounds, expected_root, exp_vs_abfe_csv


@pytest.fixture(scope="function")
def run_final(tmp_path: Path, config_and_inputs: Tuple[dict, List[str], Path, Path]) -> Dict[str, Path]:
    """
    Execute the final FEPA workflow (analysis + plotting) semantically:
      - Read existing cluster CSVs (read-only) from expected_root
      - Read exp vs ABFE CSV (read-only)
      - Generate combined_abfe_df.csv and overestimation_vs_cluster0.png under tmp_path
    Returns dict with produced artifact paths.
    """
    config, compounds, expected_root, exp_vs_abfe_csv = config_and_inputs

    out_dir = tmp_path  # All artifacts must be temporary
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Final workflow start; tmp outputs in %s", out_dir)

    # Inputs mirror the original script (read-only)
    van_list = [1]
    leg_window_list = ["coul.00", "coul.01"]

    # Load the exp vs abfe dataframe and prepare columns (read-only)
    exp_df = pd.read_csv(exp_vs_abfe_csv)
    exp_df["overestimation"] = exp_df["MBAR"] - exp_df["Experimental_G"]
    exp_df["van"] = exp_df["Vanilla"]            # semantic parity with script
    exp_df = exp_df.rename(columns={"Lig_Name": "cmp"})

    combined_df_list: List[pd.DataFrame] = []

    for cmp_name in compounds:
        cmp_expected_dir = expected_root / str(cmp_name)
        cluster_csv = cmp_expected_dir / f"{cmp_name}_conformation_cluster_df.csv"
        if not cluster_csv.exists():
            raise FileNotFoundError(f"Expected cluster CSV not found: {cluster_csv}")
        logger.info("Reading cluster CSV for %s: %s", cmp_name, cluster_csv)

        # Mirror path resolution (read-only; side-effect free)
        _ = load_abfe_paths_for_compound(
            config,
            cmp=cmp_name,
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            van_list=van_list,
            leg_window_list=leg_window_list,
            apo=False,
        )

        cluster_df = pd.read_csv(cluster_csv)

        # Extract 'van' and 'cmp' as in the script
        cluster_df["van"] = "van" + cluster_df["ensemble"].str.extract(r"van_(\d+)")
        cluster_df["cmp"] = cluster_df["ensemble"].str.extract(r"^(\d+)")[0].astype(int)

        # Merge with exp_df on cmp & van
        merged_df = cluster_df.merge(
            exp_df[["cmp", "van", "overestimation"]],
            on=["cmp", "van"],
            how="left",
        )

        # Assign sim_type and keep only ABFE rows
        merged_df["sim_type"] = merged_df["ensemble"].apply(
            lambda x: "abfe" if any(sub in x for sub in ["coul", "vdw", "rest"]) else "holo"
        )
        abfe_df = merged_df[merged_df["sim_type"] == "abfe"]

        combined_df_list.append(abfe_df)

    # Concatenate across compounds
    combined_df = pd.concat(combined_df_list, ignore_index=True)

    # Save combined dataframe to tmp_path (NOT to repository)
    combined_csv_path = out_dir / "combined_abfe_df.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info("Wrote combined CSV: %s", combined_csv_path)

    # Grouped metrics for plotting
    result = (
        combined_df.groupby(["cmp", "van"])
        .agg(
            pct_cluster0=("cluster", lambda x: (x == 0).mean() * 100),
            overestimation=("overestimation", "mean"),
        )
        .reset_index()
    )

    # Plot (non-interactive backend enforced via MPLBACKEND=Agg)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=result,
        x="pct_cluster0",
        y="overestimation",
        hue="van",
        s=100,
        ax=ax,
    )
    # Labels for cmp
    for _, row in result.iterrows():
        ax.text(row["pct_cluster0"] + 0.5, row["overestimation"], str(row["cmp"]), fontsize=9)

    # Regression line
    x = result["pct_cluster0"]
    y = result["overestimation"]
    if len(result) >= 2 and x.notna().any() and y.notna().any():
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, linestyle="--", label=f"Fit: $R^2$={r_value**2:.2f}")

    ax.set_xlabel("% rows with cluster=0")
    ax.set_ylabel("Mean overestimation")
    ax.set_title("Overestimation vs Cluster 0 occupancy")

    # Avoid legend warning: only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(title="van")
    else:
        logger.info("Skipping legend: no labeled artists present.")

    fig.tight_layout()

    plot_path = out_dir / "overestimation_vs_cluster0.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote plot: %s", plot_path)

    logger.info("Final workflow complete; artifacts: %s, %s", combined_csv_path, plot_path)
    return {"combined_csv": combined_csv_path, "plot_png": plot_path}


# --------------------------------- Main test ---------------------------------

def test_final_fepa_creates_combined_csv_and_plot(run_final: Dict[str, Path]) -> None:
    """
    End-to-end test for the final FEPA stage:
      - Reads existing inputs (read-only) from repository data.
      - Writes only to tmp_path.
      - Asserts combined CSV and plot exist (existence only).
    """
    combined_csv = run_final["combined_csv"]
    plot_png = run_final["plot_png"]

    assert combined_csv.exists(), f"Combined CSV not found: {combined_csv}"
    assert plot_png.exists(), f"Plot PNG not found: {plot_png}"

    logger.info("Verified artifacts exist: %s and %s", combined_csv, plot_png)
