"""
End-to-end pytest for FEPA side-chain torsions analysis.

This test:
  1) Loads the repo's tests/test_config/config.json
  2) Builds the minimal path_dict (van1 + coul.00/01 + apo) via FEPA utilities
  3) Runs EnsembleHandler + SideChainTorsionsFeaturizer and writes a CSV to a temp dir
  4) Reproduces the script's downstream filtering, offset, and plotting:
       - {cmp}_filtered_SideChainTorsions_features.csv
       - {cmp}_sidechain_histograms_apo.png
       - {cmp}_sidechain_evolution_apo.png
       - {cmp}_sidechain_histograms_simtype_abfe_apo.png
       - {cmp}_sidechain_histograms_simtype_holo_apo.png
  5) Verifies all artifacts exist and are non-empty
  6) Compares the temp CSV exactly to the golden CSV in tests/test_data/5_expected/1/SideChainTorsions_features.csv
"""

import logging
import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive for CI
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

# FEPA imports
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.featurizers import SideChainTorsionsFeaturizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------- helpers copied/consistent with the script ----------

def _abspath_templates(config: dict, repo_root: Path) -> dict:
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


def plot_sidechain_distribution(
    df, ensembles, output_file="ensemble_histograms.png", ncols=4, colorby="ensemble"
):
    if not isinstance(ensembles, (list, tuple, set)):
        raise ValueError("`ensembles` must be a list, tuple, or set of ensemble names")

    ensembles = list(ensembles)
    if len(ensembles) < 1:
        raise ValueError("Must provide at least one ensemble")

    filtered_df = df[df[colorby].isin(ensembles)].copy()
    torsion_columns = [col for col in df.columns if "CHI" in col]

    # JS sorting if exactly two ensembles
    if len(ensembles) == 2:
        js_scores = {}
        e1, e2 = ensembles
        for col in torsion_columns:
            d1 = filtered_df[filtered_df[colorby] == e1][col].dropna()
            d2 = filtered_df[filtered_df[colorby] == e2][col].dropna()
            bins = np.histogram_bin_edges(np.concatenate([d1, d2]), bins=30)
            p1, _ = np.histogram(d1, bins=bins, density=True)
            p2, _ = np.histogram(d2, bins=bins, density=True)
            p1 += 1e-12
            p2 += 1e-12
            p1 /= p1.sum()
            p2 /= p2.sum()
            js_scores[col] = jensenshannon(p1, p2, base=2.0)
        torsion_columns = sorted(torsion_columns, key=lambda k: js_scores[k], reverse=True)

    n_plots = len(torsion_columns)
    nrows = math.ceil(n_plots / ncols)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(torsion_columns):
        ax = axes[idx]
        sns.histplot(
            data=filtered_df,
            x=col,
            hue=colorby,
            kde=True,
            stat="density",
            common_norm=False,
            bins=30,
            ax=ax,
        )
        title = col
        if len(ensembles) == 2:
            title += f"{col}; \nJS={js_scores[col]:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # remove extra axes
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_sidechain_evolution(
    df: pd.DataFrame, ensemble_list: list, figsize=(20, 15), max_cols=4, save_path=None
):
    chi_cols = [col for col in df.columns if "CHI" in col]
    n_chis = len(chi_cols)
    ncols = min(n_chis, max_cols)
    nrows = math.ceil(n_chis / ncols)
    df_filtered = df[df["ensemble"].isin(ensemble_list)]

    palette = sns.color_palette("tab10", len(ensemble_list))
    color_dict = dict(zip(ensemble_list, palette))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i, chi in enumerate(chi_cols):
        ax = axes[i]
        for ens in ensemble_list:
            sub_df = df_filtered[df_filtered["ensemble"] == ens]
            ax.plot(sub_df["timestep"], sub_df[chi], label=ens, color=color_dict[ens])
        ax.set_title(chi, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ensemble_list), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------- pytest fixtures ----------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    # tests/a5_test/test_sc_torsions_e2e_full.py -> tests/a5_test -> tests -> REPO
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def config_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "test_config" / "config.json"


@pytest.fixture(scope="session")
def golden_csv_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "test_data" / "5_expected" / "1" / "SideChainTorsions_features.csv"


@pytest.fixture(scope="session")
def config(config_path: Path, repo_root: Path) -> dict:
    if not config_path.exists():
        pytest.skip("Test config file not found")
    cfg = load_config(str(config_path))
    return _abspath_templates(cfg, repo_root)


@pytest.fixture(scope="session")
def cmp_name(config: dict) -> str:
    # Use the first (and only) compound from the test config
    assert "compounds" in config and len(config["compounds"]) >= 1
    return str(config["compounds"][0])


@pytest.fixture()
def tmp_output_dir(tmp_path: Path, cmp_name: str) -> Path:
    out = tmp_path / "sc_torsions_end_to_end" / cmp_name
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture()
def run_pipeline_and_artifacts(config: dict, cmp_name: str, tmp_output_dir: Path):
    """Run EnsembleHandler + SCT featurizer and return all generated paths."""
    # build minimal path dict
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp_name,
        van_list=[1],
        leg_window_list=[f"coul.{i:02}" for i in range(2)],
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        apo=True,
    )

    # universes
    eh = EnsembleHandler(path_dict)
    eh.make_universes()

    # featurize
    sct = SideChainTorsionsFeaturizer(ensemble_handler=eh)
    sct.featurize()
    sct.save_features(str(tmp_output_dir), overwrite=True)

    features_csv = tmp_output_dir / "SideChainTorsions_features.csv"
    assert features_csv.exists() and features_csv.stat().st_size > 0

    # --- reproduce the downstream logic from the script ---

    features_df = pd.read_csv(features_csv)

    # offset + ROIs (kept same semantics as the script)
    offset = 567
    residues_of_interest_without_offset = [
        810, 655, 625, 628, 654, 658, 659, 809, 806, 785, 788, 802, 805, 744, 740, 781,
    ]
    residues_of_interest = [resid for resid in residues_of_interest_without_offset]

    def get_resid(colname: str):
        parts = colname.split()
        if len(parts) >= 4 and parts[3].isdigit():
            return int(parts[3]) + offset
        return None

    filtered_cols = [col for col in features_df.columns if get_resid(col) in residues_of_interest] + [
        "timestep", "ensemble"
    ]
    filtered_features_df = features_df[filtered_cols]

    def get_sim_type(ensemble: str):
        if "apo" in ensemble:
            return "apo"
        if "coul" in ensemble or "vdw" in ensemble or "rest" in ensemble:
            match = re.search(r"van_(\d+)", ensemble)
            if match:
                return "abfe"
            raise ValueError(f"van not found in ensemble: {ensemble}")
        return "holo"

    def add_offset(colname: str, _offset: int = 0):
        parts = colname.split()
        if len(parts) == 4:
            try:
                resid = int(parts[3])
                return f"{parts[0]} {parts[1]} {parts[2]} {resid + _offset}"
            except ValueError:
                return colname
        return colname

    filtered_features_df.columns = [add_offset(c, _offset=offset) for c in filtered_features_df.columns]
    filtered_csv = tmp_output_dir / f"{cmp_name}_filtered_SideChainTorsions_features.csv"
    filtered_features_df.to_csv(filtered_csv, index=False)

    # Plots
    apo_hist = tmp_output_dir / f"{cmp_name}_sidechain_histograms_apo.png"
    plot_sidechain_distribution(
        df=filtered_features_df,
        ensembles=["apo_1"],  # same as the script section kept active
        output_file=str(apo_hist),
        ncols=4,
    )

    evo_apo = tmp_output_dir / f"{cmp_name}_sidechain_evolution_apo.png"
    # NOTE: The script uses ["apo"] here; keep consistent
    plot_sidechain_evolution(
        df=filtered_features_df,
        ensemble_list=["apo"],
        figsize=(20, 15),
        max_cols=4,
        save_path=str(evo_apo),
    )

    # annotate sim_type and the two simtype plots
    filtered_features_df = filtered_features_df.copy()
    filtered_features_df["sim_type"] = filtered_features_df["ensemble"].apply(get_sim_type)

    simtype_abfe_apo = tmp_output_dir / f"{cmp_name}_sidechain_histograms_simtype_abfe_apo.png"
    plot_sidechain_distribution(
        df=filtered_features_df,
        ensembles=["abfe", "apo"],
        colorby="sim_type",
        output_file=str(simtype_abfe_apo),
        ncols=4,
    )

    simtype_holo_apo = tmp_output_dir / f"{cmp_name}_sidechain_histograms_simtype_holo_apo.png"
    plot_sidechain_distribution(
        df=filtered_features_df,
        ensembles=["holo", "apo"],
        colorby="sim_type",
        output_file=str(simtype_holo_apo),
        ncols=4,
    )

    return {
        "features_csv": features_csv,
        "filtered_csv": filtered_csv,
        "apo_hist": apo_hist,
        "evo_apo": evo_apo,
        "simtype_abfe_apo": simtype_abfe_apo,
        "simtype_holo_apo": simtype_holo_apo,
    }


# ---------- tests ----------

def test_artifacts_exist_and_nonempty(run_pipeline_and_artifacts):
    paths = run_pipeline_and_artifacts
    for key, p in paths.items():
        assert Path(p).exists(), f"{key} was not created: {p}"
        assert Path(p).stat().st_size > 0, f"{key} is empty: {p}"


def test_generated_csv_matches_golden(run_pipeline_and_artifacts, golden_csv_path: Path):
    if not golden_csv_path.exists():
        pytest.skip("Golden CSV not found at tests/test_data/5_expected/1/SideChainTorsions_features.csv")

    gen_path = run_pipeline_and_artifacts["features_csv"]
    gen_df = pd.read_csv(gen_path)
    exp_df = pd.read_csv(golden_csv_path)

    # same shape and columns
    assert gen_df.shape == exp_df.shape, f"Shape mismatch: {gen_df.shape} vs {exp_df.shape}"
    assert list(gen_df.columns) == list(exp_df.columns), "Column order/names mismatch"

    # exact equality (including floats) — mirrors your stricter test
    for col in gen_df.columns:
        if pd.api.types.is_float_dtype(gen_df[col]):
            assert np.array_equal(gen_df[col].to_numpy(), exp_df[col].to_numpy(), equal_nan=True), \
                f"Float column {col} differs from golden"
        else:
            pd.testing.assert_series_equal(gen_df[col], exp_df[col], check_names=False)


def test_basic_structure_checks(run_pipeline_and_artifacts):
    gen_df = pd.read_csv(run_pipeline_and_artifacts["features_csv"])
    assert "timestep" in gen_df.columns and "ensemble" in gen_df.columns
    chi_cols = [c for c in gen_df.columns if "CHI" in c]
    assert len(chi_cols) > 0, "Expected at least one CHI column"
    # timestep numeric & integer-like if float
    ts = gen_df["timestep"]
    assert np.issubdtype(ts.dtype, np.number)
    if np.issubdtype(ts.dtype, np.floating):
        assert np.allclose(ts.values, np.round(ts.values), atol=1e-9)
    # ensembles non-empty strings
    assert len(gen_df["ensemble"].unique()) > 0
    for e in gen_df["ensemble"].unique():
        assert isinstance(e, str) and len(e) > 0
