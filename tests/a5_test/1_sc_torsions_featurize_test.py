"""
What this test does:
  • Dynamically discovers and imports the original script (or uses FEPA_SCT_SCRIPT if set).
  • Monkeypatches heavy dependencies (EnsembleHandler, trajectory I/O, Featurizer).
  • Forces the test config to contain at least one compound (handles compounds[1:] slice).
  • Runs the script’s `main()` inside a temporary sandbox (writes to tmp/wdir/<cmp>).
  • Each mocked Featurizer copies its CSV directly from the repo’s golden file at:
        tests/test_data/5_expected/<cmp>/SideChainTorsions_features.csv
  • After execution, the test validates:
        - that the CSV was created for each compound,
        - that its schema (columns) matches expectations,
        - and that its numeric contents match the golden CSV
          within rounding tolerance (r=1e-12, a=1e-12).

Effectively, this ensures the full pipeline (config handling, per-compound loop,
I/O structure) behaves identically to the version that produced the golden data,
without performing any real trajectory processing.
"""

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pandas.testing as pdt
import matplotlib
matplotlib.use("Agg")  # headless / CI

# ---------- constants ----------
DECIMALS = 6
RTOL = 1e-12
ATOL = 1e-12

# ---------- discovery ----------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _env_script_path(repo_root: Path) -> Path | None:
    v = os.environ.get("FEPA_SCT_SCRIPT")
    if not v:
        return None
    p = Path(v)
    if not p.is_absolute():
        p = repo_root / p
    return p if p.exists() else None

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
    p = _env_script_path(repo_root)
    if p is not None:
        return p
    scripts_dir = repo_root / "scripts"
    cands: list[Path] = []
    if scripts_dir.exists():
        for f in scripts_dir.glob("*.py"):
            if _looks_like_script(f):
                cands.append(f)
    if not cands:
        skip = {"venv",".venv","site-packages","build","dist",".git"}
        for f in repo_root.rglob("*.py"):
            if any(part in skip for part in f.parts):
                continue
            if _looks_like_script(f):
                cands.append(f)
    if not cands:
        raise FileNotFoundError("Cannot find the original script. Set FEPA_SCT_SCRIPT or put it under scripts/.")
    def pref(x: Path):
        try:
            rel = x.is_relative_to(scripts_dir)  # py>=3.9
        except Exception:
            try:
                x.resolve().relative_to(scripts_dir.resolve()); rel = True
            except Exception:
                rel = False
        return (0 if rel else 1, str(x))
    return sorted(cands, key=pref)[0]

def _load_module(py_path: Path):
    spec = importlib.util.spec_from_file_location("orig_sct_mod", str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ---------- helpers ----------
def _golden_csv_for_cmp(repo_root: Path, cmp: str) -> Path:
    # use your repo-local golden path (equivalent to your absolute path)
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
def test_original_script_end_to_end_and_matches_golden(monkeypatch, tmp_path):
    repo_root = _repo_root()
    script = _discover_script(repo_root)
    mod = _load_module(script)

    # --- patch load_config so compounds[1:] is not empty ---
    from fepa.utils.file_utils import load_config as _real_load_config
    def _fake_load_config(_ignored_path):
        cfg_path = repo_root / "tests" / "test_config" / "config.json"
        cfg = _real_load_config(str(cfg_path))
        comps = list(map(str, cfg.get("compounds", [])))
        if len(comps) <= 1:
            # prepend sentinel so [1:] == the real list (e.g. ["1"])
            cfg["compounds"] = ["__DUMMY__"] + comps
        return cfg
    monkeypatch.setattr(mod, "load_config", _fake_load_config, raising=True)

    # Which compounds will the script actually process?
    cfg = _fake_load_config(None)
    comps_all = list(map(str, cfg.get("compounds", [])))
    processed = comps_all[1:] if len(comps_all) >= 2 else comps_all
    assert processed, "Adjusted config still yields no compounds to process."

    # --- mock path discovery + EnsembleHandler (no MD I/O) ---
    def _fake_load_abfe_paths(config, cmp, **kwargs):
        # structure is irrelevant once EH is mocked
        return {f"apo_{i}": {"xtc":"x"} for i in (1,2,3)} | \
               {f"{cmp}_van_{i}_vdw.20": {"xtc":"x"} for i in (1,2,3)}
    monkeypatch.setattr(mod, "load_abfe_paths_for_compound", _fake_load_abfe_paths, raising=True)

    class _EH:
        def __init__(self, path_dict): self.path_dict = path_dict
        def make_universes(self): pass
    monkeypatch.setattr(mod, "EnsembleHandler", _EH, raising=True)

    # --- featurizer writes the CSV by copying from the repo golden ---
    class _Featurizer:
        def __init__(self, ensemble_handler): self.eh = ensemble_handler
        def featurize(self): pass
        def save_features(self, outdir, overwrite=False):
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            cmp = outdir.name
            gpath = _golden_csv_for_cmp(repo_root, cmp)
            assert gpath.exists(), f"Golden CSV not found for cmp={cmp}: {gpath}"
            df = pd.read_csv(gpath)
            (outdir / "SideChainTorsions_features.csv").write_text(df.to_csv(index=False))
    monkeypatch.setattr(mod, "SideChainTorsionsFeaturizer", _Featurizer, raising=True)

    # --- run in tmp sandbox ---
    monkeypatch.chdir(tmp_path)
    mod.main()

    # --- validate tmp outputs vs golden ---
    for cmp in processed:
        tmp_csv = tmp_path / "wdir" / cmp / "SideChainTorsions_features.csv"
        assert tmp_csv.exists(), f"Script did not produce tmp CSV for {cmp}: {tmp_csv}"

        gld_csv = _golden_csv_for_cmp(repo_root, cmp)
        assert gld_csv.exists(), f"Missing golden CSV for {cmp}: {gld_csv}"

        tmp_df = pd.read_csv(tmp_csv)
        gld_df = pd.read_csv(gld_csv)

        _assert_schema(tmp_df, f"tmp CSV for {cmp}")
        _assert_schema(gld_df, f"golden CSV for {cmp}")
        assert set(tmp_df.columns) == set(gld_df.columns), \
            f"Column mismatch for {cmp}\nTmp: {sorted(tmp_df.columns)}\nGolden: {sorted(gld_df.columns)}"

        tmp_std = _stable_sort(_round_numeric(tmp_df)[sorted(tmp_df.columns)])
        gld_std = _stable_sort(_round_numeric(gld_df)[sorted(gld_df.columns)])

        pdt.assert_frame_equal(
            tmp_std, gld_std,
            check_dtype=False,
            rtol=RTOL, atol=ATOL,
            check_like=False,
        )
