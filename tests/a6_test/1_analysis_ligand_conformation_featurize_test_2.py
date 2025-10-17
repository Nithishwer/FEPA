# tests/test_fepa_integration.py
"""
End-to-end FEPA integration test that semantically reproduces the original script's
behavior using FEPA's public APIs (no script import/execution; no verbatim retype).

Guarantees:
- No persistent files: all outputs are written only under pytest's tmp_path.
- No __pycache__ / .pyc: disable bytecode generation at import time.
- Warnings silenced globally and at test level.
- Inputs are read-only from the same locations the original script expects.
- One main test compares our temporary CSV against the repository's expected CSV,
  and asserts a temporary PDB centroid file exists.
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
import pandas.testing as pdt
import pytest
import mdaencore as encore

# FEPA 
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.utils.md_utils import check_bp_residue_consistency

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    # Absolutize relative templates in-memory.
    for k in ("abfe_window_path_template", "vanilla_path_template", "vanilla_path_template_old", "apo_path_template"):
        if k in cfg:
            p = Path(cfg[k])
            if not p.is_absolute():
                cfg[k] = str((repo_root / p).resolve())

    compounds: List[str] = list(cfg["compounds"])
    if not compounds:
        raise RuntimeError("No compounds found in FEPA test config.")
    compounds.sort()
    first_cmp = compounds[0]

    expected_root = repo_root / "tests" / "test_data" / "6_expected"
    logger.info("First compound=%s; expected_root=%s", first_cmp, expected_root)
    return cfg, first_cmp, expected_root


@pytest.fixture(scope="function")
def run_e2e(tmp_path: Path, config_and_first_cmp: Tuple[dict, str, Path]) -> Dict[str, Path]:
    """
    Execute an end-to-end run using FEPA, writing outputs
    strictly under tmp_path. Returns outputs paths and the reference CSV path.
    """
    config, first_cmp, expected_root = config_and_first_cmp
    tmp_out = tmp_path / first_cmp
    tmp_out.mkdir(parents=True, exist_ok=True)

    # Inputs mirror the script's choices (read-only)
    van_list = [1]
    leg_window_list = ["coul.00", "coul.01"]

    # Resolve ABFE/holo paths for this compound
    path_dict = load_abfe_paths_for_compound(
        config,
        cmp=first_cmp,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        van_list=van_list,
        leg_window_list=leg_window_list,
        apo=False,
    )
    logger.info("Built path_dict for %s", first_cmp)

    # Build universes
    eh = EnsembleHandler(path_dict)
    eh.make_universes()
    uni_all = eh.get_universe_dict()
    check_bp_residue_consistency(uni_all)

    # Keep holo-only 
    uni = {k: u for k, u in uni_all.items() if "apo" not in k.lower()}
    if not uni:
        raise RuntimeError("No holo universes available after filtering.")

    # Helper: infer ligand resname if config value is missing/wrong.
    def _infer_ligand_resname(universes: Dict[str, object]) -> str | None:
        COMMON = {
            "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
            "SOL","HOH","TIP3","NA","K","CL","CLA","POT","MG","CA","POPC","POPE","POPS","CHL1","CHOL","DOPC","DPPC",
        }
        counts: Dict[str, int] = {}
        for u in universes.values():
            for res in u.residues:
                rn = str(res.resname).upper()
                if rn in COMMON:
                    continue
                ag = res.atoms.select_atoms("not name H*")
                if ag.n_atoms >= 6:
                    counts[rn] = counts.get(rn, 0) + 1
        return max(counts, key=counts.get) if counts else None

    # Prefer config-provided ligand resname; else infer.
    lig_resname = (config.get("ligand_resname") or "").strip().upper()
    if not lig_resname:
        lig_resname = _infer_ligand_resname(uni) or "UNK"

    # Try selection with preferred/inferred name; if absent, attempt inference fallback once.
    def _filter_universes_by_lig(u_dict: Dict[str, object], resname: str) -> Tuple[Dict[str, object], str]:
        sel = f"resname {resname} and not name H*"
        keep = {k: u for k, u in u_dict.items() if u.select_atoms(sel).n_atoms > 0}
        return keep, sel

    uni_try, lig_sel = _filter_universes_by_lig(uni, lig_resname)
    if not uni_try:
        inferred = _infer_ligand_resname(uni)
        if inferred and inferred != lig_resname:
            lig_resname = inferred
            uni_try, lig_sel = _filter_universes_by_lig(uni, lig_resname)

    if not uni_try:
        raise RuntimeError(f"No holo universes contain a detectable ligand (tried '{lig_resname}').")

    # Deterministic order and global frame mapping
    keys = list(uni_try.keys())
    ens = [uni_try[k] for k in keys]
    lengths = [len(u.trajectory) for u in ens]
    if any(n == 0 for n in lengths):
        raise RuntimeError("One or more universes have zero frames.")
    offsets = np.cumsum([0] + lengths[:-1])
    total = int(sum(lengths))
    ensemble_series = [None] * total
    timestep_series = [None] * total
    for i, key in enumerate(keys):
        off = int(offsets[i])
        for j in range(lengths[i]):
            ensemble_series[off + j] = key
            timestep_series[off + j] = j

    # Cluster ligand conformations (mdaencore)
    dbscan = encore.DBSCAN(eps=0.5, min_samples=5, algorithm="auto", leaf_size=30)
    clusters = encore.cluster(
        ensembles=ens,
        select=lig_sel,
        superimposition_subset="name CA",
        method=dbscan,
    )

    # Assign cluster IDs per frame and dump a centroid to tmp_path
    cluster_series = [None] * total
    first_centroid_pdb = None
    for cl in clusters:
        cid = int(cl.id)
        for gi in cl.elements:
            cluster_series[int(gi)] = cid

        centroid_gid = int(cl.centroid)
        centroid_ensemble = ensemble_series[centroid_gid]
        centroid_timestep = int(timestep_series[centroid_gid])
        out_pdb = (tmp_out / f"{first_cmp}_conformation_cluster_{cid}.pdb").resolve()
        eh.dump_frames(ensemble=centroid_ensemble, timestep=centroid_timestep, save_path=str(out_pdb))
        if first_centroid_pdb is None:
            first_centroid_pdb = out_pdb

    # Persist cluster assignment CSV to tmp_path
    df = pd.DataFrame({"timestep": timestep_series, "ensemble": ensemble_series, "cluster": cluster_series})
    prod_csv = (tmp_out / f"{first_cmp}_conformation_cluster_df.csv").resolve()
    df.to_csv(prod_csv, index=False)
    logger.info("Wrote CSV: %s", prod_csv)
    if first_centroid_pdb:
        logger.info("Wrote PDB: %s", first_centroid_pdb)

    # Reference CSV 
    ref_csv = (expected_root / first_cmp / f"{first_cmp}_conformation_cluster_df.csv").resolve()

    return {
        "csv_path": prod_csv,
        "pdb_path": first_centroid_pdb,
        "ref_csv_path": ref_csv,
        "compound": Path(first_cmp),
    }


@pytest.mark.filterwarnings("ignore::Warning")
def test_e2e_csv_matches_and_pdb_exists(run_e2e: Dict[str, Path]) -> None:
    """
    Produce temporary CSV & PDB via FEPA and verify:
      - CSV exactly matches the repository's expected CSV 
      - PDB centroid file exists under tmp_path
    """
    prod_csv = run_e2e["csv_path"]
    ref_csv = run_e2e["ref_csv_path"]
    pdb_path = run_e2e["pdb_path"]

    # Sanity: required files must exist 
    assert prod_csv and prod_csv.exists(), f"Produced CSV not found: {prod_csv}"
    assert ref_csv and ref_csv.exists(), f"Reference CSV not found: {ref_csv}"

    # Load and compare; drop obviously nondeterministic cols only if present 
    df_prod = pd.read_csv(prod_csv)
    df_ref = pd.read_csv(ref_csv)
    for col in ("timestamp", "datetime", "uuid"):
        if col in df_prod.columns and col in df_ref.columns:
            df_prod = df_prod.drop(columns=[col])
            df_ref = df_ref.drop(columns=[col])

    pdt.assert_frame_equal(df_prod, df_ref, check_like=False, check_dtype=False)
    logger.info("CSV comparison passed for %s", run_e2e["compound"])

    # PDB existence
    assert pdb_path is not None and pdb_path.exists(), f"PDB artifact not found: {pdb_path}"
    logger.info("PDB existence check passed for %s", pdb_path)
