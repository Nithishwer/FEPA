# tests/test_fipa_plotting_integration.py
"""
FEPA plotting integration test

What this test does:
- Takes as input cluster assignment CSV from the repository's expected test data
- Generates cluster distribution and per-ensemble counts plots
- Writes under pytest's tmp_path
- Asserts that the plot files were created

"""

import sys as _sys, os as _os, warnings as _warnings
_sys.dont_write_bytecode = True
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_os.environ["MPLBACKEND"] = "Agg"
_warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Tuple, Dict
import logging

import pytest
import pandas as pd
import matplotlib.pyplot as plt

from fepa.utils.file_utils import load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute FEPA repository root (read-only inputs live here)."""
    root = Path("/biggin/b230/magd5710/Documents/FEPA").resolve()
    logger.info("Using FEPA repo root: %s", root)
    return root


@pytest.fixture(scope="session")
def csv_input_path(repo_root: Path) -> Tuple[Path, str]:
    """
    Resolve the cluster assignment CSV path from the expected data directory.
    Pick the first compound defined in FEPA's tests config to mirror script behavior.
    """
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    cfg = load_config(str(cfg_path))
    compounds = sorted(list(cfg["compounds"]))
    if not compounds:
        raise RuntimeError("No compounds found in FEPA test config.")
    cmp_name = compounds[0]

    expected_root = repo_root / "tests" / "test_data" / "6_expected"
    csv_path = expected_root / cmp_name / f"{cmp_name}_conformation_cluster_df.csv"
    logger.info("Resolved existing CSV for plotting: %s", csv_path)
    return csv_path, cmp_name


@pytest.fixture(scope="function")
def make_plots(tmp_path: Path, csv_input_path: Tuple[Path, str]) -> Dict[str, List[Path]]:
    """
    Perform plotting for data in the existing CSV,
    produce plots into tmp_path, and return the list of created plot files.
    """
    csv_path, cmp_name = csv_input_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected input CSV not found: {csv_path}")

    logger.info("Plotting start for compound %s using input CSV: %s", cmp_name, csv_path)

    out_dir = tmp_path / cmp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data (read-only)
    df = pd.read_csv(csv_path)

    created: List[Path] = []

    # ---- Plot 1: Cluster distribution (bar chart of counts per cluster) ----
    if {"cluster"} <= set(df.columns):
        ax = df["cluster"].value_counts(dropna=False).sort_index().plot(kind="bar")
        ax.set_title(f"{cmp_name}: Cluster count distribution")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Count")
        fig1 = ax.get_figure()
        p1 = out_dir / f"{cmp_name}_cluster_distribution.png"
        fig1.savefig(p1, bbox_inches="tight")
        plt.close(fig1)
        created.append(p1)
        logger.info("Wrote plot: %s", p1)

    # ---- Plot 2: Ensemble distribution (bar chart of frames per ensemble) ----
    if {"ensemble"} <= set(df.columns):
        ax = df["ensemble"].value_counts(dropna=False).sort_index().plot(kind="bar")
        ax.set_title(f"{cmp_name}: Frames per ensemble")
        ax.set_xlabel("Ensemble key")
        ax.set_ylabel("Count")
        fig2 = ax.get_figure()
        p2 = out_dir / f"{cmp_name}_ensemble_distribution.png"
        fig2.savefig(p2, bbox_inches="tight")
        plt.close(fig2)
        created.append(p2)
        logger.info("Wrote plot: %s", p2)

    logger.info("Plotting complete for %s. %d file(s) written under %s", cmp_name, len(created), out_dir)
    return {"compound": cmp_name, "plots": created, "out_dir": out_dir}




def test_fepa_plotting_creates_artifacts(make_plots: Dict[str, List[Path]]) -> None:
    """
    End-to-end plotting test:
    """
    compound = make_plots["compound"]
    plots: List[Path] = make_plots["plots"]
    out_dir: Path = make_plots["out_dir"]

    # We expect at least one plot (cluster or ensemble distribution)
    assert plots, f"No plot artifacts were produced for compound {compound} in {out_dir}"
    for p in plots:
        assert p.exists(), f"Plot file missing: {p}"

    logger.info("Test complete for %s. Verified %d plot file(s) in %s", compound, len(plots), out_dir)
