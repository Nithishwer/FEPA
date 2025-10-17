# tests/test_fepa_integration.py
"""
End-to-end FEPA integration test that REPRODUCES the original script's behavior
*directly in this test file* by calling the same FEPA APIs and libraries.

Mirrored from the original script:
- Loads FEPA tests config at /biggin/b230/magd5710/Documents/FEPA/tests/test_config/config.json
- Uses load_abfe_paths_for_compound with van_list=[1], leg_window_list=["coul.00","coul.01"], apo=False
- Builds universes via EnsembleHandler, checks binding-pocket residue consistency
- Infers ligand resname (config override else heuristic, else "UNK"), selects heavy atoms
- Clusters ligand conformations with mdaencore.DBSCAN, dumps centroid frames to PDB
- Writes cluster assignment CSV "{cmp}_conformation_cluster_df.csv"

Reference test data path (derived identically to the script):
  repo_root = Path("../../FEPA")
  expected CSV: repo_root / "tests/test_data/6_expected" / <cmp> / f"{cmp}_conformation_cluster_df.csv"

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import mdaencore as encore

from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.utils.md_utils import check_bp_residue_consistency

# Module-level logger and constants (only two functions below).
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Common non-ligand residue names excluded when inferring a ligand resname
_COMMON_NON_LIG = {
    # protein
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",

    # solvent/ions/lipids
    "SOL",
    "HOH",
    "TIP3",
    "NA",
    "K",
    "CL",
    "CLA",
    "POT",
    "MG",
    "CA",
    "POPC",
    "POPE",
    "POPS",
    "CHL1",
    "CHOL",
    "DOPC",
    "DPPC",
}


def run_pipeline(output_dir: Path) -> Dict[str, Path]:
    """
    Reproduce the original pipeline end-to-end, but write all outputs
    (CSV + PDBs) into `output_dir` (per-compound subdirs). Inputs are
    read from the exact same locations as in the original script.

    Returns:
        {
          "csv_path": Path to produced CSV for the first compound,
          "pdb_path": Path to one produced PDB centroid for the first compound,
          "ref_csv_path": Path to reference CSV under tests/test_data/6_expected,
          "compound": The first compound name processed
        }
    """
    logger.info("Starting FEPA pipeline reproduction...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Inputs identical to original script (READ-ONLY) ---
    repo_root = Path("/biggin/b230/magd5710/Documents/FEPA").resolve()
    config_path = repo_root / "tests" / "test_config" / "config.json"
    expected_root = repo_root / "tests" / "test_data" / "6_expected"

    # Load config and absolutize templates in-memory (same keys as original)
    config = load_config(str(config_path))
    for k in [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]:
        if k in config:
            p = Path(config[k])
            if not p.is_absolute():
                config[k] = str((repo_root / p).resolve())

    van_list = [1]
    leg_window_list = ["coul.00", "coul.01"]

    compounds: List[str] = list(config["compounds"])
    if not compounds:
        raise RuntimeError("No compounds found in FEPA test config.")
    # Process in stable order and use the first compound for return values
    compounds.sort()
    first_cmp = compounds[0]

    produced_csv_path: Path | None = None
    produced_pdb_path: Path | None = None

    for cmp_name in compounds:
        cmp_out = (output_dir / cmp_name).resolve()
        cmp_out.mkdir(parents=True, exist_ok=True)

        # Build paths for FEPA ensembles
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp=cmp_name,
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            van_list=van_list,
            leg_window_list=leg_window_list,
            apo=False,
        )
        logger.info("Built path_dict for %s", cmp_name)

        # Create universes and validate binding-pocket residues
        eh = EnsembleHandler(path_dict)
        eh.make_universes()
        uni_dict_all = eh.get_universe_dict()
        check_bp_residue_consistency(uni_dict_all)

        # Holo universes only (guard as in original)
        uni_dict = {k: u for k, u in uni_dict_all.items() if "apo" not in k.lower()}
        if not uni_dict:
            raise RuntimeError("No holo universes available. Check test inputs.")

        # Ligand resname: config override -> heuristic -> "UNK"
        lig_resname = (config.get("ligand_resname") or "").strip().upper()
        if not lig_resname:
            counts = {}
            for u in uni_dict.values():
                for res in u.residues:
                    rn = str(res.resname).upper()
                    if rn in _COMMON_NON_LIG:
                        continue
                    ag = res.atoms.select_atoms("not name H*")
                    if ag.n_atoms >= 6:
                        counts[rn] = counts.get(rn, 0) + 1
            lig_resname = max(counts, key=counts.get) if counts else "UNK"

        lig_sel = f"resname {lig_resname} and not name H*"
        # Drop universes lacking the ligand selection
        drop = [k for k, u in uni_dict.items() if u.select_atoms(lig_sel).n_atoms == 0]
        for k in drop:
            uni_dict.pop(k, None)
        if not uni_dict:
            raise RuntimeError(f"No holo universes contain ligand '{lig_resname}'.")

        # Stable ordering drives global frame indexing
        keys_in_order = list(uni_dict.keys())
        ensembles = [uni_dict[k] for k in keys_in_order]

        # Build global index mapping (concatenated frame indices)
        traj_lengths = [len(u.trajectory) for u in ensembles]
        if any(n == 0 for n in traj_lengths):
            raise RuntimeError("One or more universes have zero frames.")
        offsets = np.cumsum([0] + traj_lengths[:-1])
        total_frames = int(sum(traj_lengths))
        ensemble_series = [None] * total_frames
        timestep_series = [None] * total_frames
        for i, key in enumerate(keys_in_order):
            n = traj_lengths[i]
            off = int(offsets[i])
            for j in range(n):
                ensemble_series[off + j] = key
                timestep_series[off + j] = j

        # Cluster ligand conformations; dump centroid frames (to tmp_path)
        dbscan_method = encore.DBSCAN(
            eps=0.5, min_samples=5, algorithm="auto", leaf_size=30
        )
        cluster_collection = encore.cluster(
            ensembles=ensembles,
            select=lig_sel,
            superimposition_subset="name CA",
            method=dbscan_method,
        )

        cluster_series = [None] * total_frames
        first_centroid_pdb_for_cmp: Path | None = None
        for clus in cluster_collection:
            cluster_id = int(clus.id)
            indices = list(clus.elements)  # global indices
            for idx in indices:
                cluster_series[idx] = cluster_id

            centroid_gid = int(clus.centroid)
            centroid_ensemble = ensemble_series[centroid_gid]
            centroid_timestep = int(timestep_series[centroid_gid])
            out_pdb = (
                cmp_out / f"{cmp_name}_conformation_cluster_{cluster_id}.pdb"
            ).resolve()
            eh.dump_frames(
                ensemble=centroid_ensemble,
                timestep=centroid_timestep,
                save_path=str(out_pdb),
            )
            if first_centroid_pdb_for_cmp is None:
                first_centroid_pdb_for_cmp = out_pdb

        # Persist assignments CSV (to tmp_path)
        cluster_df = pd.DataFrame(
            {
                "timestep": timestep_series,
                "ensemble": ensemble_series,
                "cluster": cluster_series,
            }
        )
        out_csv = (cmp_out / f"{cmp_name}_conformation_cluster_df.csv").resolve()
        cluster_df.to_csv(out_csv, index=False)
        logger.info("Wrote CSV for %s: %s", cmp_name, out_csv)
        if first_centroid_pdb_for_cmp is not None:
            logger.info("Wrote PDB for %s: %s", cmp_name, first_centroid_pdb_for_cmp)

        # Capture the first compound's artifacts for return
        if cmp_name == first_cmp:
            produced_csv_path = out_csv
            produced_pdb_path = first_centroid_pdb_for_cmp

    # Reference CSV path for the first compound (as per original script; READ-ONLY)
    ref_csv_path = (
        expected_root / first_cmp / f"{first_cmp}_conformation_cluster_df.csv"
    ).resolve()

    if produced_csv_path is None or produced_pdb_path is None:
        raise RuntimeError(
            "Failed to produce expected artifacts for the first compound."
        )

    logger.info("Pipeline completed for first compound: %s", first_cmp)
    return {
        "csv_path": produced_csv_path,
        "pdb_path": produced_pdb_path,
        "ref_csv_path": ref_csv_path,
        "compound": Path(first_cmp),
    }


def test_end_to_end_csv_and_pdb(tmp_path: Path) -> None:
    """
    Single end-to-end test:
      - Runs the reproduced pipeline writing to tmp_path.
      - Compares produced CSV (tmp_path) to reference CSV (repo tests/test_data/6_expected) via assert_frame_equal.
      - Asserts a PDB centroid file exists under tmp_path.
      - Logs major steps.
    """
    logger.info("Test start: running FEPA pipeline into %s", tmp_path)

    # Execute pipeline with outputs confined to tmp_path
    artifacts = run_pipeline(tmp_path)

    # Load and compare CSVs exactly (drop only obviously nondeterministic columns if present)
    prod_csv = artifacts["csv_path"]
    ref_csv = artifacts["ref_csv_path"]
    assert prod_csv.exists(), f"Produced CSV not found: {prod_csv}"
    assert ref_csv.exists(), f"Reference CSV not found: {ref_csv}"

    df_prod = pd.read_csv(prod_csv)
    df_ref = pd.read_csv(ref_csv)

    pdt.assert_frame_equal(df_prod, df_ref, check_like=False, check_dtype=False)
    logger.info("CSV comparison passed for %s", artifacts["compound"])

    # PDB check
    pdb_path = artifacts["pdb_path"]
    assert pdb_path is not None and pdb_path.exists(), (
        f"PDB artifact not found under tmp_path: {pdb_path}"
    )

    logger.info("Test end: FEPA pipeline integration successful.")
