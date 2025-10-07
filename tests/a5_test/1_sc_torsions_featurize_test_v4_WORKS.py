import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"MDAnalysis\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*hydrogenbonds.*")

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SideChainTorsionsFeaturizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_schema(df: pd.DataFrame):
    need = {"timestep", "ensemble"}
    assert need.issubset(df.columns), f"missing columns: {need - set(df.columns)}"
    assert any(str(c).startswith("CHI") for c in df.columns), "no CHI* columns"


def _count_frames(tpr: Path, xtc: Path) -> int:
    import MDAnalysis as mda
    u = mda.Universe(str(tpr), str(xtc))
    return int(u.trajectory.n_frames)


def _filter_existing_nonempty_pairs(path_dict: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    kept: dict[str, dict[str, str]] = {}
    for ens, d in path_dict.items():
        tpr = Path(d.get("tpr", ""))
        xtc = Path(d.get("xtc", ""))
        try:
            if tpr.exists() and xtc.exists() and tpr.stat().st_size > 0 and xtc.stat().st_size > 0:
                kept[ens] = d
        except OSError:
            pass
    return kept


@pytest.mark.slow
def test_sc_torsions_featurize_selfcontained(tmp_path, caplog):
    """
    Self-contained integration test for side-chain torsion featurization.
    Uses only tests/test_config/config.json and tests/test_data/1 (coul.00/coul.01 + optional apo).
    Writes ONLY to tmp_path.
    """
    repo = _repo_root()
    cfg_path = repo / "tests" / "test_config" / "config.json"
    data_base = repo / "tests" / "test_data" / "1"
    assert cfg_path.exists(), f"missing test config: {cfg_path}"
    assert (data_base / "vanilla_rep_1" / "npt.gro").exists(), "missing vanilla_rep_1/npt.gro"

    # Load config
    config = load_config(str(cfg_path))
    compounds = list(config.get("compounds", []))
    assert compounds, "test config has no 'compounds'"
    cmp_name = compounds[0]

    # Choose a coul leg that exists
    leg_selected = None
    for leg in ("coul.00", "coul.01"):
        prod = data_base / leg / "prod"
        if (prod / "prod.tpr").exists() or (prod / "topol.tpr").exists():
            if (prod / "prod.xtc").exists() or (prod / "traj_comp.xtc").exists():
                leg_selected = leg
                break
    assert leg_selected, "no leg with tpr+xtc under (coul.00|coul.01)/prod"

    # Build path dict using FEPA helper
    with caplog.at_level("INFO"):
        logging.getLogger(__name__).info("Loading paths for compound %s ...", cmp_name)
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp_name,
            van_list=[1],                    # minimal
            leg_window_list=[leg_selected],  # coul.00 or coul.01
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

    # Make all paths absolute and PRESERVE metadata (e.g., bp_selection_string)
    abspath_path_dict: dict[str, dict[str, str]] = {}
    default_bp = "name CA and resid " + config["pocket_residues_string"]
    for ens, d in path_dict.items():
        newd = dict(d)  # preserve everything (bp_selection_string, labels, etc.)
        tpr = Path(newd.get("tpr", ""))
        xtc = Path(newd.get("xtc", ""))
        if not tpr.is_absolute():
            tpr = (repo / tpr).resolve()
        if not xtc.is_absolute():
            xtc = (repo / xtc).resolve()
        newd["tpr"] = str(tpr)
        newd["xtc"] = str(xtc)
        # Ensure the selection string exists (belt-and-braces)
        if not newd.get("bp_selection_string"):
            newd["bp_selection_string"] = default_bp
        abspath_path_dict[ens] = newd

    # Filter to usable pairs
    abspath_path_dict = _filter_existing_nonempty_pairs(abspath_path_dict)
    assert abspath_path_dict, "no usable (tpr, xtc) pairs after filtering"

    # Build universes
    with caplog.at_level("INFO"):
        logging.getLogger(__name__).info("Loading trajectories for compound %s ...", cmp_name)
        eh = EnsembleHandler(abspath_path_dict)
        eh.make_universes()

    # Featurize
    with caplog.at_level("INFO"):
        logging.getLogger(__name__).info("Featurizing side-chain torsions ...")
        sct = SideChainTorsionsFeaturizer(ensemble_handler=eh)
        sct.featurize()

    # Save ONLY under tmp_path
    out_dir = tmp_path / "wdir_tmp" / cmp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with caplog.at_level("INFO"):
        logging.getLogger(__name__).info("Saving features for compound %s ...", cmp_name)
        sct.save_features(str(out_dir), overwrite=True)

    out_csv = out_dir / "SideChainTorsions_features.csv"
    assert out_csv.exists(), f"missing CSV: {out_csv}"
    assert out_csv.stat().st_size > 0, "features CSV is empty"

    # Validate schema & values
    df = pd.read_csv(out_csv)
    _assert_schema(df)
    chi_cols = [c for c in df.columns if str(c).startswith("CHI")]
    assert chi_cols, "no CHI* columns"
    assert np.isfinite(df[chi_cols].to_numpy()).all(), "non-finite values present in CHI*"

    # Frame parity & monotonicity
    ens_to_paths: dict[str, tuple[Path, Path]] = {}
    apo_tpr = data_base / "apo" / "prod.tpr"
    apo_xtc = data_base / "apo" / "prod.xtc"
    if apo_tpr.exists() and apo_xtc.exists():
        ens_to_paths["apo_1"] = (apo_tpr, apo_xtc)

    prod = data_base / leg_selected / "prod"
    tpr = prod / "prod.tpr" if (prod / "prod.tpr").exists() else prod / "topol.tpr"
    xtc = prod / "prod.xtc" if (prod / "prod.xtc").exists() else prod / "traj_comp.xtc"
    if tpr.exists() and xtc.exists():
        ens_to_paths[f"1_van_1_{leg_selected}"] = (tpr, xtc)

    for ens, sub in df.groupby("ensemble"):
        if ens not in ens_to_paths:
            continue
        ts = sub["timestep"].to_numpy()
        assert (ts[1:] >= ts[:-1]).all(), f"non-monotonic timestep for {ens}"
        tpr_e, xtc_e = ens_to_paths[ens]
        if tpr_e.exists() and xtc_e.exists() and tpr_e.stat().st_size > 0 and xtc_e.stat().st_size > 0:
            n_frames = _count_frames(tpr_e, xtc_e)
            assert len(sub) == n_frames, f"rows ({len(sub)}) != frames ({n_frames}) for {ens}"

    # Logs
    blob = " ".join(r.message for r in caplog.records)
    assert "Loading paths for compound" in blob
    assert "Loading trajectories" in blob
    assert "Saving features" in blob
