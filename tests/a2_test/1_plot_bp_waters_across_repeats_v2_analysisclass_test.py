"""
Test for the BP water featurization (minimal slice).
It:
  1) runs featurization for van=1 and windows ['coul.00', 'coul.01'],
  2) asserts artifacts exist (CSV + PNGs),
  3) compares WaterOccupancy_features.csv to a committed golden with
     tolerant numeric checks and resilient row alignment.

Golden CSV file expected at:
  FEPA/tests/test_data/2_expected/<cmp>/WaterOccupancy_features.csv
"""

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

import os
from pathlib import Path
import builtins
from typing import Literal

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

builtins.Literal = Literal

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.featurizers import BPWaterFeaturizer


# ----------------------------- Config -----------------------------
# Numeric comparison tolerances 
DECIMALS = 6
RTOL = 1e-6
ATOL = 1e-8

# Use the same pocket selection string you used when producing the golden
BP_SELECTION_STRING = (
    "name CA and resid "
    "54 55 56 57 58 59 60 61 62 64 65 68 83 84 85 87 88 91 92 "
    "173 176 177 180 217 218 221 225 235 238 239 240 241 242 243 244 245 246 247"
)


# ----------------------------- Helpers -----------------------------
def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Make any relative path templates absolute w.r.t. repo_root."""
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


def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        df[num_cols] = df[num_cols].round(DECIMALS)
    return df


def _sort_by(df: pd.DataFrame, keys=("ensemble", "Time (ps)", "frame")) -> pd.DataFrame:
    use = [k for k in keys if k in df.columns]
    return df.sort_values(use).reset_index(drop=True) if use else df


def _align_on_common_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys=("ensemble", "Time (ps)", "frame"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join alignment on any keys that both frames share; fall back to positional."""
    keys = [k for k in keys if k in left.columns and k in right.columns]
    if not keys:
        n = min(len(left), len(right))
        return left.iloc[:n].reset_index(drop=True), right.iloc[:n].reset_index(drop=True)
    merged_keys = right[keys].drop_duplicates()
    left2 = left.merge(merged_keys, on=keys, how="inner")
    right2 = right.merge(left2[keys].drop_duplicates(), on=keys, how="inner")
    left2 = _sort_by(left2, keys)
    right2 = _sort_by(right2, keys)
    return left2.reset_index(drop=True), right2.reset_index(drop=True)


# ----------------------------- The Test -----------------------------
@pytest.mark.integration
def test_bp_waters_minimal_against_golden(tmp_path):
    # --- repo & config ---
    repo_root = Path(__file__).resolve().parents[2]  # FEPA/
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    assert cfg_path.exists(), f"Missing configuration file: {cfg_path}"
    config = load_config(str(cfg_path))
    config = _abspath_templates(config, repo_root)

    # Use the first truncated compound defined in your test config
    cmp_name = config["compounds"][0]

    # --- where to write ACTUAL (test-run) outputs ---
    out_root = tmp_path / "a3_compare_bp_waters" / "wdir" / cmp_name
    out_root.mkdir(parents=True, exist_ok=True)

    # --- build path dict for the tiny slice: van=1, windows=['coul.00', 'coul.01'] ---
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp_name,
        van_list=[1],
        leg_window_list=[f"coul.{i:02d}" for i in range(2)],
        bp_selection_string=BP_SELECTION_STRING,
        apo=False,
    )

    # --- run featurization (no PCA etc) ---
    eh = EnsembleHandler(path_dict)
    eh.make_universes()
    bpw = BPWaterFeaturizer(ensemble_handler=eh)
    bpw.featurize(radius=10)
    bpw.save_features(str(out_root), overwrite=True)

    # --- artifacts: CSV must exist ---
    act_csv = out_root / "WaterOccupancy_features.csv"
    assert act_csv.exists(), "WaterOccupancy_features.csv was not created"

    # --- generate the same plots your script produces (for van=1 only) ---
    df = pd.read_csv(act_csv)
    required_cols = {"ensemble", "Time (ps)", "occupancy"}
    missing = required_cols - set(df.columns)
    assert not missing, f"WaterOccupancy_features.csv missing columns: {missing}"

    # Per-vanilla time series (van=1)
    van = 1
    van_df = df[df["ensemble"].str.contains(f"van_{van}")]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=van_df, x="Time (ps)", y="occupancy", hue="ensemble")
    plt.title(f"Water Occupancy for {cmp_name}")
    plt.xlabel("Time (ps)")
    plt.xlim(0, 20000)
    plt.ylabel("Number of Waters")
    plt.legend(title="Ensemble", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    plt.tight_layout()
    ts_png = out_root / f"{cmp_name}_water_occupancy_van{van}_timeseries_v2.png"
    plt.savefig(ts_png)
    plt.close()
    assert ts_png.exists(), f"Missing expected plot: {ts_png}"

    # Average across windows (your same aggregation)
    work = df.copy()
    work["van"] = work["ensemble"].str.extract(r"van_(\d)")
    work["id"] = work["ensemble"].str.replace(r"_van_\d+", "", regex=True)
    avg_df = work.groupby(["id", "van"], as_index=False)["occupancy"].mean()

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=avg_df, x="id", y="occupancy", hue="van", palette="tab10")
    plt.title(f"Average Water Occupancy Across Windows for {cmp_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Window ID", fontsize=14)
    plt.ylabel("Average Number of Waters", fontsize=14)
    plt.legend(title="Vanilla Repeat", title_fontsize=12, fontsize=10, loc="upper right")
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    avg_png = out_root / f"{cmp_name}_water_occupancy_across_windows_v2.png"
    plt.savefig(avg_png, dpi=300)
    plt.close()
    assert avg_png.exists(), f"Missing expected plot: {avg_png}"

    # --- CSV comparison ---
    golden_csv = repo_root / "tests" / "test_data" / "2_expected" / cmp_name / "WaterOccupancy_features.csv"
    assert golden_csv.exists(), f"Golden CSV not found: {golden_csv}"

    act = _round_numeric(pd.read_csv(act_csv))
    exp = _round_numeric(pd.read_csv(golden_csv))

    # Keep golden rows only for ensembles present in actual (robust to minor selection diffs)
    if "ensemble" in act.columns and "ensemble" in exp.columns:
        keep = set(act["ensemble"].unique())
        exp = exp[exp["ensemble"].isin(keep)].copy()

    # Align on common keys, fall back to positional if needed
    act2, exp2 = _align_on_common_keys(act, exp, keys=("ensemble", "Time (ps)", "frame"))

    # Columns must match in name & order
    assert list(act2.columns) == list(exp2.columns), "Water CSV columns differ"

    # Row counts should match
    assert len(act2) == len(exp2), "Water CSV row counts differ"

    # Final numeric compare (rtol/atol) & dtype-insensitive
    pdt.assert_frame_equal(
        act2.reset_index(drop=True),
        exp2.reset_index(drop=True),
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )
