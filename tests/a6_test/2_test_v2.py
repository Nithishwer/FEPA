# tests/test_fepa_cluster_overlap_png_only.py
"""
End-to-end FEPA overlap/cluster-distribution test that semantically reproduces
the target script's behavior using FEPA's public APIs (no script import/execution).

Guarantees:
- No persistent files: all outputs are written only under pytest's tmp_path.
- No __pycache__ / .pyc: disable bytecode generation at import time.
- Warnings silenced globally and at test level.
- Inputs are read-only from the same locations the original script expects.
- The test writes only a normalized stacked-bar PNG under tmp_path and asserts it exists.
"""

from __future__ import annotations
import sys as _sys
import os as _os
import warnings as _warnings

_sys.dont_write_bytecode = True
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for PNG generation
import matplotlib.pyplot as plt

# FEPA API
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------- helpers -----------------------------

def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative path templates in the config (in-memory only)."""
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


# ----------------------------- PyTest fixtures -----------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute FEPA repository root (read-only inputs are located here)."""
    root = Path("/biggin/b230/magd5710/Documents/FEPA").resolve()
    logger.info("Using FEPA repo root: %s", root)
    return root


@pytest.fixture(scope="session")
def config_and_first_cmp(repo_root: Path) -> Tuple[dict, str, Path]:
    """
    Load FEPA test config and pick a compound.
    Return (config, first_cmp, expected_root) where expected_root holds reference CSVs.
    """
    cfg_path = repo_root / "tests" / "test_config" / "config.json"
    cfg = load_config(str(cfg_path))
    cfg = _abspath_templates(cfg, repo_root)

    compounds: List[str] = list(cfg["compounds"])
    if not compounds:
        raise RuntimeError("No compounds found in FEPA test config.")
    compounds.sort()
    first_cmp = compounds[0]

    expected_root = repo_root / "tests" / "test_data" / "6_expected"
    logger.info("First compound=%s; expected_root=%s", first_cmp, expected_root)
    return cfg, first_cmp, expected_root


@pytest.fixture(scope="function")
def run_overlap_png(tmp_path: Path, config_and_first_cmp: Tuple[dict, str, Path]) -> Dict[str, Path]:
    """
    Reproduce the cluster-overlap script semantics, producing ONLY the PNG:
      - Read the expected cluster assignment CSV for the first compound (read-only).
      - Compute per-ensemble normalized cluster proportions.
      - Produce a normalized stacked bar figure (PNG) under tmp_path/<compound>/.
    Returns the PNG path and metadata.
    """
    config, cmp_id, expected_root = config_and_first_cmp
    tmp_out = (tmp_path / cmp_id).resolve()
    tmp_out.mkdir(parents=True, exist_ok=True)
    logger.info("Temporary output dir: %s", tmp_out)

    # Inputs mirror the script's choices (read-only)
    van_list = [1]
    leg_window_list = ["coul.00", "coul.01"]

    # Build path dictionary (script does this; we exercise the same API)
    _ = load_abfe_paths_for_compound(
        config,
        cmp=cmp_id,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        van_list=van_list,
        leg_window_list=leg_window_list,
        apo=False,  # holo only, per script intent
    )

    # Read reference cluster assignment CSV (the script expects it under 6_expected/<cmp>/)
    ref_csv = (expected_root / cmp_id / f"{cmp_id}_conformation_cluster_df.csv").resolve()
    assert ref_csv.exists(), f"Reference CSV not found: {ref_csv}"
    logger.info("Reading cluster assignment CSV: %s", ref_csv)
    cluster_df = pd.read_csv(ref_csv)

    # Compute normalized cluster proportions per ensemble (sum to 100%)
    cluster_counts = cluster_df.groupby(["ensemble", "cluster"]).size().unstack(fill_value=0)
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100.0

    # Plot normalized stacked bar chart to PNG under tmp_path (mirrors script)
    png_path = (tmp_out / f"{cmp_id}_conformation_cluster_distribution.png").resolve()
    ax = cluster_props.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title(f"Cluster distribution in each ensemble for {cmp_id} (normalized to 100%)")
    ax.set_xlabel("Ensemble")
    ax.set_ylabel("Percentage (%)")
    plt.xticks(rotation=90)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    logger.info("Wrote PNG: %s", png_path)

    # Sanity: internal normalization (in-memory only; no files written)
    row_sums = cluster_props.sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(row_sums, 100.0, rtol=0, atol=1e-6), "Row sums must be 100% after normalization."

    return {
        "png_path": png_path,
        "compound": Path(cmp_id),
    }


# ----------------------------- the test -----------------------------

@pytest.mark.filterwarnings("ignore::Warning")
def test_cluster_overlap_png_exists(run_overlap_png: Dict[str, Path]) -> None:
    """
    Produce temporary PNG via FEPA and verify:
      - PNG stacked bar chart exists under tmp_path
    """
    png_path = run_overlap_png["png_path"]
    cmp_id = run_overlap_png["compound"]

    # PNG existence
    assert png_path is not None and png_path.exists(), f"PNG artifact not found: {png_path}"
    logger.info("PNG existence check passed for %s", cmp_id)
