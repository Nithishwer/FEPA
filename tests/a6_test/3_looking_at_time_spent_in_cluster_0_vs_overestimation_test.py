# tests/a6_test/3_looking_at_time_spent_in_cluster_0_vs_overestimation_test.py
"""
- Reads existing inputs only (no writes outside pytest's tmp_path)
- Creates a combined CSV and a PNG plot in tmp_path
- Asserts output files exist
- Also compares the produced combined CSV against the repo's expected combined CSV
"""

from __future__ import annotations

# ---- Guards: disable bytecode, silence warnings, non-GUI matplotlib BEFORE other imports ----
import sys as _sys, os as _os, warnings as _warnings

_sys.dont_write_bytecode = True
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_os.environ["MPLBACKEND"] = "Agg"
_warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# FEPA public APIs (read-only usage)
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import (
    load_abfe_paths_for_compound,
)  # used to mirror path resolution (no writes)
from fepa.core.ensemble_handler import (
    EnsembleHandler,
)  # imported to match original semantics (no direct use here)

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------------- Path helpers --------------------------------


def _detect_repo_root() -> Path:
    """
    Compute the repo root relative to this test file (or via FEPA_REPO_ROOT if provided).
    """
    env_root = _os.getenv("FEPA_REPO_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.is_dir():
            return p

    here = Path(__file__).resolve()
    for anc in here.parents:
        if (anc / "tests").is_dir():
            return anc
    # Fallback (common layout: <repo_root>/tests/<subdir>/<file>.py)
    return here.parents[2]


def _abspath_templates(cfg: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative templates in the FEPA test config (in-memory only)."""
    out = dict(cfg)
    for k in (
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ):
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out


def _find_exp_vs_abfe_csv(repo_root: Path) -> Path:
    """
    Find the experimental vs ABFE CSV.
    """

    # 1) Optional environment override (useful for CI or custom layouts)
    env_path = _os.getenv("FEPA_EXP_VS_ABFE_CSV")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()

    # 2) Common repo-relative candidates
    candidates = [
        repo_root / "tests" / "test_data" / "exp_v_abfe_df_van_reps_all_hrex.csv",
        repo_root
        / "tests"
        / "test_data"
        / "6_expected"
        / "exp_v_abfe_df_van_reps_all_hrex.csv",
        repo_root
        / "tests"
        / "a6_test"
        / "test_data"
        / "exp_v_abfe_df_van_reps_all_hrex.csv",
        repo_root / "test_data" / "exp_v_abfe_df_van_reps_all_hrex.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()

    # 3) Fallback: targeted recursive search under tests/
    tests_root = repo_root / "tests"
    if tests_root.exists():
        for p in tests_root.rglob("exp_v_abfe_df_van_reps_all_hrex.csv"):
            if p.is_file():
                return p.resolve()

    looked = "\n  - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        "Exp vs ABFE CSV not found in any repo-relative location. Looked for:\n"
        f"  - {looked}\n"
        "Also searched recursively under <repo_root>/tests/.\n"
        "Please place or symlink the file into one of the above paths, or set FEPA_EXP_VS_ABFE_CSV."
    )


# ----------------------------- Fixtures (read-only inputs) -----------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    root = _detect_repo_root()
    logger.info("Using FEPA repo root (relative): %s", root)
    return root


@pytest.fixture(scope="session")
def config_and_inputs(repo_root: Path) -> Tuple[dict, List[str], Path, Path, Path]:
    """
    Load FEPA test config, absolutize templates in-memory, and return:
      (config, compounds, expected_root, exp_vs_abfe_csv_path, ref_combined_csv)
    All paths resolved **relative to repo_root**.
    """
    cfg_path = (repo_root / "tests" / "test_config" / "config.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_config(str(cfg_path))
    cfg = _abspath_templates(cfg, repo_root)

    compounds = sorted(list(cfg["compounds"]))
    if not compounds:
        raise RuntimeError("No compounds found in FEPA test config.")

    expected_root = (repo_root / "tests" / "test_data" / "6_expected").resolve()
    if not expected_root.exists():
        raise FileNotFoundError(f"Expected data root not found: {expected_root}")

    # Reference combined CSV produced previously and checked into repo
    ref_combined_csv = (expected_root / "combined_abfe_df.csv").resolve()
    if not ref_combined_csv.exists():
        raise FileNotFoundError(f"Reference combined CSV not found: {ref_combined_csv}")

    exp_vs_abfe_csv = _find_exp_vs_abfe_csv(repo_root)

    logger.info(
        "Resolved expected_root=%s; exp_vs_abfe_csv=%s; ref_combined_csv=%s",
        expected_root,
        exp_vs_abfe_csv,
        ref_combined_csv,
    )
    return cfg, compounds, expected_root, exp_vs_abfe_csv, ref_combined_csv


@pytest.fixture(scope="function")
def run_final(
    tmp_path: Path, config_and_inputs: Tuple[dict, List[str], Path, Path, Path]
) -> Dict[str, Path]:
    """
      - Read existing cluster CSVs from expected_root
      - Read exp vs ABFE CSV
      - Generate combined_abfe_df.csv and overestimation_vs_cluster0.png under tmp_path
    Returns dict with produced files paths and reference paths.
    """
    config, compounds, expected_root, exp_vs_abfe_csv, ref_combined_csv = (
        config_and_inputs
    )

    out_dir = tmp_path  # All artifacts must be temporary
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Final workflow start; tmp outputs in %s", out_dir)

    # Inputs mirror the original script (read-only)
    van_list = [1]
    leg_window_list = ["coul.00", "coul.01"]

    # Load the exp vs abfe dataframe and prepare columns (read-only)
    exp_df = pd.read_csv(exp_vs_abfe_csv)
    # Expect columns: ["MBAR", "Experimental_G", "Vanilla", "Lig_Name", ...]
    exp_df["overestimation"] = exp_df["MBAR"] - exp_df["Experimental_G"]
    exp_df["van"] = exp_df["Vanilla"]  # semantic parity with script
    exp_df = exp_df.rename(columns={"Lig_Name": "cmp"})

    combined_df_list: List[pd.DataFrame] = []

    for cmp_name in compounds:
        # Location of cluster CSV produced earlier in pipeline (golden truth / expected)
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

        # Extract 'van' and 'cmp' to match original semantics
        # 'ensemble' usually encodes "...van_<N>..." and starts with "<cmp>_..."
        cluster_df["van"] = "van" + cluster_df["ensemble"].str.extract(r"van_(\d+)")[0]
        cluster_df["cmp"] = cluster_df["ensemble"].str.extract(r"^(\d+)")[0].astype(int)

        # Merge with exp_df on cmp & van
        merged_df = cluster_df.merge(
            exp_df[["cmp", "van", "overestimation"]],
            on=["cmp", "van"],
            how="left",
        )

        # Assign sim_type and keep only ABFE rows
        merged_df["sim_type"] = merged_df["ensemble"].apply(
            lambda x: "abfe"
            if any(sub in x for sub in ["coul", "vdw", "rest"])
            else "holo"
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
        ax.text(
            row["pct_cluster0"] + 0.5,
            row["overestimation"],
            str(row["cmp"]),
            fontsize=9,
        )

    # Regression line (only if we have >= 2 points)
    x = result["pct_cluster0"]
    y = result["overestimation"]
    if len(result) >= 2 and x.notna().any() and y.notna().any():
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_fit = np.linspace(float(x.min()), float(x.max()), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, linestyle="--", label=f"Fit: $R^2$={r_value**2:.2f}")

    ax.set_xlabel("% rows with cluster=0")
    ax.set_ylabel("Mean overestimation")
    ax.set_title("Overestimation vs Cluster 0 occupancy")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(title="van")

    fig.tight_layout()

    plot_path = out_dir / "overestimation_vs_cluster0.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info("Wrote plot: %s", plot_path)

    logger.info(
        "Final workflow complete; artifacts: %s, %s", combined_csv_path, plot_path
    )
    return {
        "combined_csv": combined_csv_path,
        "plot_png": plot_path,
        "ref_combined_csv": ref_combined_csv,
    }


# --------------------------------- Main test ---------------------------------


def _align_for_compare(
    df_prod: pd.DataFrame, df_ref: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align columns and deterministically sort rows for a robust equality check.
    - Keep only common columns
    - Reorder produced columns to match reference order
    - Sort both by common sort keys or, if absent, by all column names
    """
    common_cols = [c for c in df_ref.columns if c in df_prod.columns]
    df_prod = df_prod[common_cols].copy()
    df_ref = df_ref[common_cols].copy()

    sort_keys = [
        c for c in ["cmp", "van", "ensemble", "timestep", "cluster"] if c in common_cols
    ]
    if not sort_keys:
        sort_keys = sorted(common_cols)

    df_prod = df_prod.sort_values(by=sort_keys).reset_index(drop=True)
    df_ref = df_ref.sort_values(by=sort_keys).reset_index(drop=True)
    return df_prod, df_ref


def test_final_fepa_creates_combined_csv_and_plot(run_final: Dict[str, Path]) -> None:
    """
    End-to-end test for the final FEPA stage:
      - Reads existing inputs (read-only) from repository data
      - Writes only to tmp_path
      - Asserts combined CSV and plot exist
      - Compares the produced combined CSV against the expected combined CSV
    """
    combined_csv = run_final["combined_csv"]
    plot_png = run_final["plot_png"]
    ref_combined_csv = run_final["ref_combined_csv"]

    # Existence checks
    assert combined_csv.exists(), f"Combined CSV not found: {combined_csv}"
    assert plot_png.exists(), f"Plot PNG not found: {plot_png}"
    assert ref_combined_csv.exists(), (
        f"Reference combined CSV not found: {ref_combined_csv}"
    )

    # Load and compare CSVs
    df_prod = pd.read_csv(combined_csv)
    df_ref = pd.read_csv(ref_combined_csv)

    df_prod, df_ref = _align_for_compare(df_prod, df_ref)

    # Allow dtype differences and column order differences (already aligned), exact row equality required
    pdt.assert_frame_equal(df_prod, df_ref, check_dtype=False, check_like=True)

    logger.info(
        "Verified artifacts and CSV equality: %s and %s", combined_csv, plot_png
    )
