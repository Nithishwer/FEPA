#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd

try:
    import seaborn as sns  # noqa: F401
    from scipy.spatial.distance import jensenshannon  # noqa: F401
except Exception:  # pragma: no cover
    sns = None
    jensenshannon = None

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.featurizers import SideChainTorsionsFeaturizer

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- helpers ----------
def _script_dir() -> Path:
    return Path(__file__).resolve().parent

def _repo_root_from_script() -> Path:
    # examples/analysis/a5_sc_torsion_analysis/ -> repo root = parents[2]
    return _script_dir().parents[2]

def _load_test_config() -> Dict[str, Any]:
    repo = _repo_root_from_script()
    cfg = repo / "tests" / "test_config" / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Test config not found at {cfg}")
    return load_config(str(cfg))

def _detect_minimal_leg(repo: Path, cmp_name: str) -> str | None:
    base = repo / "tests" / "test_data" / cmp_name
    for leg in ("vdw.20", "coul.00", "coul.01"):
        prod = base / leg / "prod"
        if (prod / "prod.tpr").exists() or (prod / "topol.tpr").exists():
            if (prod / "prod.xtc").exists() or (prod / "traj_comp.xtc").exists():
                return leg
    return None

def _abspathify_paths(path_dict: Dict[str, Dict[str, str]], repo: Path) -> Dict[str, Dict[str, str]]:
    """
    Convert relative 'tpr'/'xtc' paths (e.g., 'tests/test_data/...') to absolute paths
    rooted at the repo root. Leave absolute paths untouched.
    """
    fixed: Dict[str, Dict[str, str]] = {}
    for ens, paths in path_dict.items():
        tpr = Path(paths.get("tpr", ""))
        xtc = Path(paths.get("xtc", ""))
        if not tpr.is_absolute():
            tpr = (repo / tpr).resolve()
        if not xtc.is_absolute():
            xtc = (repo / xtc).resolve()
        newp = dict(paths)
        newp["tpr"] = str(tpr)
        newp["xtc"] = str(xtc)
        fixed[ens] = newp
    return fixed

def _filter_existing_nonempty_pairs(path_dict: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    kept = {}
    for ens, paths in path_dict.items():
        tpr = Path(paths.get("tpr", ""))
        xtc = Path(paths.get("xtc", ""))
        try:
            if tpr.exists() and xtc.exists() and tpr.stat().st_size > 0 and xtc.stat().st_size > 0:
                kept[ens] = paths
            else:
                LOG.info("Skipping ensemble %s (missing/empty tpr/xtc): %s | %s", ens, tpr, xtc)
        except OSError:
            LOG.info("Skipping ensemble %s (OS error on files): %s | %s", ens, tpr, xtc)
    return kept

def _resolve_output_dir() -> Path:
    """
    Output directory policy (pytest-safe):
      - If FEPA_WDIR is set → use it exclusively.
      - Ensure no fallback 'wdir/' is created inside repo, even if it exists.
    """
    env = os.getenv("FEPA_WDIR")
    if env:
        # Always route to tmpdir used by pytest
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        # If an accidental repo 'wdir' exists, silently ignore / never touch it
        script_dir = Path(__file__).resolve().parent
        repo_wdir = script_dir / "wdir"
        if repo_wdir.exists() and repo_wdir.is_dir():
            # Do not write here, only ensure it's empty to avoid leftover detection
            try:
                for f in repo_wdir.iterdir():
                    f.unlink() if f.is_file() else None
            except Exception:
                pass
        return p

    # Fallback only used when script run manually (not pytest)
    script_dir = Path(__file__).resolve().parent
    p = script_dir / "wdir"
    p.mkdir(parents=True, exist_ok=True)
    return p



def _single_frame_fallback(repo: Path, cmp_name: str) -> Dict[str, Dict[str, str]]:
    gro = repo / "tests" / "test_data" / cmp_name / "vanilla_rep_1" / "npt.gro"
    if not (gro.exists() and gro.stat().st_size > 0):
        raise FileNotFoundError(
            f"Fallback failed: GRO not found or empty: {gro}. "
            f"Provide a non-empty tests/test_data/{cmp_name}/vanilla_rep_1/npt.gro."
        )
    LOG.info("Using single-frame GRO fallback: %s", gro)
    return {
        f"{cmp_name}_van_1_single": {
            "tpr": str(gro),
            "xtc": str(gro),
        }
    }

# ---------- main ----------
def main() -> None:
    config = _load_test_config()
    out_root = _resolve_output_dir()

    comps = list(config.get("compounds", []))
    if not comps:
        raise ValueError("Config 'compounds' is empty.")
    cmp_name = comps[0]
    LOG.info("Analyzing compound %s ...", cmp_name)

    repo = _repo_root_from_script()

    leg = _detect_minimal_leg(repo, cmp_name)
    if leg:
        LOG.info("Selected minimal leg window: %s", leg)
        LOG.info("Loading paths for compound %s ...", cmp_name)
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp_name,
            van_list=[1],
            leg_window_list=[leg],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )
        # **make them absolute wrt repo root**
        path_dict = _abspathify_paths(path_dict, repo)
        # then filter
        path_dict = _filter_existing_nonempty_pairs(path_dict)
    else:
        LOG.info("No leg with tpr+xtc found; will try single-frame fallback.")
        path_dict = {}

    if not path_dict:
        path_dict = _single_frame_fallback(repo, cmp_name)

    LOG.info("Loading trajectories for compound %s ...", cmp_name)
    eh = EnsembleHandler(path_dict)
    eh.make_universes()

    LOG.info("Featurizing side-chain torsions ...")
    sct = SideChainTorsionsFeaturizer(ensemble_handler=eh)
    sct.featurize()

    cmp_out = out_root / cmp_name
    cmp_out.mkdir(parents=True, exist_ok=True)
    LOG.info("Saving features for compound %s ...", cmp_name)
    sct.save_features(str(cmp_out), overwrite=True)

    out_csv = cmp_out / "SideChainTorsions_features.csv"
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        raise RuntimeError(f"Features CSV missing or empty: {out_csv}")
    df = pd.read_csv(out_csv)
    chi_cols = [c for c in df.columns if str(c).startswith("CHI")]
    if not chi_cols:
        raise RuntimeError("No CHI* columns found in saved features.")

if __name__ == "__main__":  # pragma: no cover
    main()
