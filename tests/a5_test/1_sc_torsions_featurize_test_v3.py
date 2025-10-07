import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"MDAnalysis\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*hydrogenbonds.*")

import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sct_script_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _assert_schema(df: pd.DataFrame):
    need = {"timestep", "ensemble"}
    assert need.issubset(df.columns), f"missing columns: {need - set(df.columns)}"
    assert any(str(c).startswith("CHI") for c in df.columns), "no CHI* columns"


def _count_frames(tpr: Path, xtc: Path) -> int:
    import MDAnalysis as mda
    u = mda.Universe(str(tpr), str(xtc))
    return int(u.trajectory.n_frames)


@pytest.mark.slow
def test_sc_torsions_script_fullstack(tmp_path, caplog, monkeypatch):
    repo = _repo_root()

    # 1) Script + config presence
    script = repo / "examples" / "analysis" / "a5_sc_torsion_analysis" / "1_sc_torsions_featurize_v2.py"
    assert script.exists(), f"script not found: {script}"
    assert (repo / "tests" / "test_config" / "config.json").exists(), "missing tests/test_config/config.json"

    # 2) Minimal data availability
    data_base = repo / "tests" / "test_data" / "1"
    assert (data_base / "vanilla_rep_1" / "npt.gro").exists(), "missing vanilla_rep_1/npt.gro"
    ok_leg = False
    for leg in ("coul.00", "coul.01"):
        prod = data_base / leg / "prod"
        if (prod / "prod.tpr").exists() or (prod / "topol.tpr").exists():
            if (prod / "prod.xtc").exists() or (prod / "traj_comp.xtc").exists():
                ok_leg = True
                break
    assert ok_leg, "no leg with tpr+xtc found under tests/test_data/1/(coul.00|coul.01)/prod"

    # 3) Load real module
    mod = _load_module(script)

    # Confirm real FEPA classes wired
    from fepa.core.ensemble_handler import EnsembleHandler as RealEH
    from fepa.core.featurizers import SideChainTorsionsFeaturizer as RealSCT
    assert getattr(mod, "EnsembleHandler", None) is RealEH
    assert getattr(mod, "SideChainTorsionsFeaturizer", None) is RealSCT

    # 4) Run from the script dir; route outputs to tmp
    script_dir = script.parent
    os.chdir(script_dir)

    # single source of truth for output dir
    wdir_tmp = tmp_path / "wdir_tmp"
    monkeypatch.setenv("FEPA_WDIR", str(wdir_tmp))

    # >>> sandbox the script: force its resolver to return our tmp dir
    monkeypatch.setattr(mod, "_resolve_output_dir", lambda: wdir_tmp)

    with caplog.at_level("INFO"):
        mod.main()

    # 5) Validate outputs (tmp only)
    out_csv = wdir_tmp / "1" / "SideChainTorsions_features.csv"
    assert out_csv.exists(), f"missing CSV: {out_csv}"
    assert out_csv.stat().st_size > 0, "features CSV is empty"

    df = pd.read_csv(out_csv)
    _assert_schema(df)
    chi_cols = [c for c in df.columns if str(c).startswith("CHI")]
    assert chi_cols, "no CHI* columns"
    assert np.isfinite(df[chi_cols].to_numpy()).all(), "non-finite values present in CHI*"

    # 6) Frame parity & monotonicity (coul.* + optional apo)
    ens_to_paths: dict[str, tuple[Path, Path]] = {}
    apo_tpr = data_base / "apo" / "prod.tpr"
    apo_xtc = data_base / "apo" / "prod.xtc"
    if apo_tpr.exists() and apo_xtc.exists():
        ens_to_paths["apo_1"] = (apo_tpr, apo_xtc)

    for leg in ("coul.00", "coul.01"):
        prod = data_base / leg / "prod"
        tpr = prod / "prod.tpr" if (prod / "prod.tpr").exists() else prod / "topol.tpr"
        xtc = prod / "prod.xtc" if (prod / "prod.xtc").exists() else prod / "traj_comp.xtc"
        if tpr.exists() and xtc.exists():
            ens_to_paths[f"1_van_1_{leg}"] = (tpr, xtc)
            break

    for ens, sub in df.groupby("ensemble"):
        if ens not in ens_to_paths:
            continue
        ts = sub["timestep"].to_numpy()
        assert (ts[1:] >= ts[:-1]).all(), f"non-monotonic timestep for {ens}"
        tpr, xtc = ens_to_paths[ens]
        if tpr.exists() and xtc.exists() and tpr.stat().st_size > 0 and xtc.stat().st_size > 0:
            n_frames = _count_frames(tpr, xtc)
            assert len(sub) == n_frames, f"rows ({len(sub)}) != frames ({n_frames}) for {ens}"

    # 7) Logging sanity
    blob = " ".join(r.message for r in caplog.records)
    assert "Loading paths for compound" in blob
    assert "Loading trajectories" in blob
    assert "Saving features" in blob

    # 8) Ensure repo wasn't dirtied
    assert not (script_dir / "wdir").exists(), "Repo was dirtied; outputs must go to FEPA_WDIR only."
