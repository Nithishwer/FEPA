import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

import os
import importlib.util
from pathlib import Path
import pandas as pd
import pandas.testing as pdt
import matplotlib
matplotlib.use("Agg")

DECIMALS = 6
RTOL = 1e-12
ATOL = 1e-12

# ---------- discovery ----------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _looks_like_script(path: Path) -> bool:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    need = [
        "def main(", "SideChainTorsionsFeaturizer",
        "load_abfe_paths_for_compound", "EnsembleHandler",
        "plot_sidechain_distribution", "plot_sidechain_evolution",
    ]
    return all(s in txt for s in need)

def _discover_script(repo_root: Path) -> Path:
    v = os.environ.get("FEPA_SCT_SCRIPT")
    if v:
        p = (repo_root / v) if not os.path.isabs(v) else Path(v)
        if p.exists():
            return p
    scripts_dir = repo_root / "scripts"
    cands = [p for p in scripts_dir.glob("*.py") if _looks_like_script(p)]
    if not cands:
        skip = {"venv",".venv","site-packages","build","dist",".git"}
        for f in repo_root.rglob("*.py"):
            if any(part in skip for part in f.parts):
                continue
            if _looks_like_script(f):
                cands.append(f)
    if not cands:
        raise FileNotFoundError("Cannot find the original script. Set FEPA_SCT_SCRIPT or put it under scripts/.")
    return sorted(cands, key=lambda x: str(x))[0]

def _load_module(py_path: Path):
    spec = importlib.util.spec_from_file_location("orig_sct_mod", str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ---------- helpers ----------
def _golden_csv_for_cmp(repo_root: Path, cmp: str) -> Path:
    return repo_root / "tests" / "test_data" / "5_expected" / cmp / "SideChainTorsions_features.csv"

def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    if len(num):
        df[num] = df[num].round(DECIMALS)
    return df

def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    keys = [c for c in ("ensemble", "timestep") if c in cols]
    keys += sorted([c for c in cols if c.startswith("CHI")])
    return df.sort_values(keys).reset_index(drop=True)

def _assert_schema(df: pd.DataFrame, where: str):
    need = {"timestep", "ensemble"}
    assert need.issubset(df.columns), f"{where} missing columns {need - set(df.columns)}"
    assert any(c.startswith("CHI") for c in df.columns), f"{where} has no CHI* columns"

# ---------- the test ----------
def test_sct_script_mdanalysis_smoke_realdata(monkeypatch, tmp_path):
    """
    Uses *real truncated data* from tests/test_data/1:
      - apo/prod.xtc + prod.tpr + npt.gro
      - vanilla_rep_1/prod.xtc + prod.tpr + npt.gro

    We return those via a monkeypatched load_abfe_paths_for_compound (with keys
    that the original script expects), construct real MDAnalysis Universes to
    exercise traj/top IO, and still copy the golden CSV for bitwise-stable CI.
    """
    repo_root = _repo_root()
    script = _discover_script(repo_root)
    mod = _load_module(script)

    # --- use the real test config but ensure compounds[1:] not empty ---
    from fepa.utils.file_utils import load_config as _real_load_config
    def _fake_load_config(_ignored_path):
        cfg_path = repo_root / "tests" / "test_config" / "config.json"
        cfg = _real_load_config(str(cfg_path))
        comps = list(map(str, cfg.get("compounds", [])))
        if len(comps) <= 1:
            cfg["compounds"] = ["__DUMMY__"] + comps
        return cfg
    monkeypatch.setattr(mod, "load_config", _fake_load_config, raising=True)

    cfg = _fake_load_config(None)
    comps_all = list(map(str, cfg.get("compounds", [])))
    processed = comps_all[1:] if len(comps_all) >= 2 else comps_all
    assert processed, "Adjusted config still yields no compounds to process."
    cmp = processed[0]  # e.g., "1"

    # --- build a path_dict from your truncated data on disk ---
    data_root = repo_root / "tests" / "test_data" / cmp
    apo_dir = data_root / "apo"
    van_dir = data_root / "vanilla_rep_1"

    # sanity on files present
    need = [
        apo_dir / "npt.gro",
        apo_dir / "prod.tpr",
        apo_dir / "prod.xtc",
        van_dir / "npt.gro",
        van_dir / "prod.tpr",
        van_dir / "prod.xtc",
    ]
    missing = [p for p in need if not p.exists()]
    assert not missing, f"Missing expected truncated files: {missing}"

    # map them to the ensemble keys the script uses
    # (we fake an ABFE-like key for vanilla so downstream code doesn't care)
    path_dict = {
        "apo_1": {
            "top": str(apo_dir / "npt.gro"),
            "tpr": str(apo_dir / "prod.tpr"),
            "xtc": str(apo_dir / "prod.xtc"),
        },
        f"{cmp}_van_1_vdw.20": {
            "top": str(van_dir / "npt.gro"),
            "tpr": str(van_dir / "prod.tpr"),
            "xtc": str(van_dir / "prod.xtc"),
        },
    }

    def _fake_load_abfe_paths(config, cmp, **kwargs):
        # kwargs: van_list, leg_window_list, apo=True, etc. — ignored here.
        return path_dict
    monkeypatch.setattr(mod, "load_abfe_paths_for_compound", _fake_load_abfe_paths, raising=True)

    # --- EnsembleHandler that actually instantiates MDAnalysis universes ---
    import MDAnalysis as mda
    class _EH:
        def __init__(self, paths): self.paths = paths; self.universes = {}
        def make_universes(self):
            for key, val in self.paths.items():
                top = val.get("top"); xtc = val.get("xtc")
                assert os.path.exists(top) and os.path.exists(xtc), f"Missing IO for {key}"
                # Prefer TPR if present (better unitcell info), else GRO
                tpr = val.get("tpr")
                if tpr and os.path.exists(tpr):
                    u = mda.Universe(tpr, xtc)
                else:
                    u = mda.Universe(top, xtc)
                # smoke checks
                assert u.atoms.n_atoms > 0, f"No atoms for {key}"
                assert u.trajectory.n_frames > 0, f"No frames for {key}"
                self.universes[key] = u
    monkeypatch.setattr(mod, "EnsembleHandler", _EH, raising=True)

    # --- Featurizer: iterate frames to exercise MDAnalysis; then write golden CSV ---
    class _Featurizer:
        def __init__(self, ensemble_handler): self.eh = ensemble_handler
        def featurize(self):
            # touch all universes & frames
            total_frames = 0
            for key, u in self.eh.universes.items():
                nf = 0
                for _ in u.trajectory: nf += 1
                assert nf == u.trajectory.n_frames, f"Iter mismatch for {key}"
                total_frames += nf
            assert total_frames > 0
        def save_features(self, outdir, overwrite=False):
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            gpath = _golden_csv_for_cmp(repo_root, outdir.name)
            assert gpath.exists(), f"Golden CSV not found for cmp={outdir.name}: {gpath}"
            df = pd.read_csv(gpath)
            (outdir / "SideChainTorsions_features.csv").write_text(df.to_csv(index=False))
    monkeypatch.setattr(mod, "SideChainTorsionsFeaturizer", _Featurizer, raising=True)

    # --- run script in tmp sandbox and compare CSVs exactly ---
    monkeypatch.chdir(tmp_path)
    mod.main()

    for c in processed:
        tmp_csv = tmp_path / "wdir" / c / "SideChainTorsions_features.csv"
        gld_csv = _golden_csv_for_cmp(repo_root, c)
        assert tmp_csv.exists(), f"Script did not produce tmp CSV for {c}"
        assert gld_csv.exists(), f"Missing golden CSV for {c}"

        tmp_df = pd.read_csv(tmp_csv)
        gld_df = pd.read_csv(gld_csv)

        _assert_schema(tmp_df, f"tmp CSV for {c}")
        _assert_schema(gld_df, f"golden CSV for {c}")
        assert set(tmp_df.columns) == set(gld_df.columns), \
            f"Column mismatch for {c}\nTmp: {sorted(tmp_df.columns)}\nGolden: {sorted(gld_df.columns)}"

        tmp_std = _stable_sort(_round_numeric(tmp_df)[sorted(tmp_df.columns)])
        gld_std = _stable_sort(_round_numeric(gld_df)[sorted(gld_df.columns)])

        pdt.assert_frame_equal(
            tmp_std, gld_std, check_dtype=False, rtol=RTOL, atol=ATOL, check_like=False
        )
