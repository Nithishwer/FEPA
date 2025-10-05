"""
Validates BOTH:
  A) per-compound aggregation CSV:
       FEPA/tests/test_data/2_expected/<cmp>/avg_water_occupancy.csv
     against an aggregation recomputed from:
       FEPA/tests/test_data/2_expected/<cmp>/WaterOccupancy_features.csv

  B) global correlation CSV:
       FEPA/tests/test_data/2_expected/water_occupancy_vs_delG.csv
     against a correlation recomputed from:
       FEPA/tests/test_data/2_expected/<cmp>/WaterOccupancy_features.csv
       FEPA/tests/test_data/2_expected/exp_vs_abfe_min.csv

- No MD is run; only CSVs are read.
- Temporary artifacts (for debugging) are written to pytest's tmp_path.
- Robust to ensemble label formats and dtype mismatches on keys.
"""

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

from pathlib import Path
import numpy as np
import pandas as pd
import pandas.testing as pdt


# --------------------- Tolerances ---------------------
DECIMALS = 6
RTOL = 1e-6
ATOL = 1e-8


# --------------------- Helpers (parsing & types) ---------------------
def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    if len(num):
        df[num] = df[num].round(DECIMALS)
    return df

def _sort_by(df: pd.DataFrame, keys) -> pd.DataFrame:
    use = [k for k in keys if k in df.columns]
    return df.sort_values(use).reset_index(drop=True) if use else df

def _extract_window(series: pd.Series) -> pd.Series:
    """Find vdw.NN / rest.NN / coul.NN anywhere in the string."""
    return series.str.extract(r"(vdw\.\d+|rest\.\d+|coul\.\d+)", expand=False)

def _extract_van(series: pd.Series) -> pd.Series:
    """Extract vanilla number from 'van_1' or 'van-1' anywhere in the string."""
    return series.str.extract(r"van[_-]?(\d+)", expand=False)

def _normalize_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce schema to canonical types so merges don’t fail on dtype."""
    out = df.copy()
    if "id" in out.columns:
        out["id"] = out["id"].astype(str)
    if "van" in out.columns:
        out["van"] = out["van"].astype(str).str.replace(".0", "", regex=False)
    if "occupancy" in out.columns:
        out["occupancy"] = pd.to_numeric(out["occupancy"], errors="coerce")
    return out

def _normalize_corr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("compound", "window_id", "van"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ("occupancy_selected", "exp_delG", "pred_delG", "Overestimation"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# --------------------- Aggregation recompute ---------------------
def _recompute_avg_from_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute avg occupancy by (id, van) from the features file."""
    w = features_df.copy()
    w["id"]  = _extract_window(w["ensemble"])
    w["van"] = _extract_van(w["ensemble"])
    w = w.dropna(subset=["id", "van"])
    assert not w.empty, "Parsed features are empty after extracting (id, van)."
    out = w.groupby(["id", "van"], as_index=False)["occupancy"].mean()
    return _normalize_agg(out)

def _choose_window_id(avg_df: pd.DataFrame, preferred: str | None) -> str:
    """Pick a usable window id from avg_df['id']."""
    ids = avg_df["id"].dropna().astype(str)
    if preferred and (ids == preferred).any():
        return preferred
    # prefer max vdw, then rest, then coul; else first
    parsed = ids.str.extract(r"^(vdw|rest|coul)\.(\d+)$")
    if not parsed.isna().all().all():
        tmp = pd.DataFrame({"id": ids})
        tmp["_fam"] = parsed[0]
        tmp["_idx"] = pd.to_numeric(parsed[1], errors="coerce")
        tmp = tmp.dropna(subset=["_fam", "_idx"])
        for fam in ("vdw", "rest", "coul"):
            sub = tmp[tmp["_fam"] == fam]
            if not sub.empty:
                return f"{fam}.{int(sub['_idx'].max()):02d}"
    return ids.iloc[0]


# --------------------- Correlation recompute ---------------------
def _recompute_correlation_from_features_and_exp(
    expected_root: Path,
    preferred_window: str | None = None,
    use_saved_window_if_present: bool = True,
) -> pd.DataFrame:
    """
    Build the correlation table by:
      1) recomputing avg occupancy per compound from features,
      2) choosing a window id (prefer saved one if available),
      3) joining to exp_vs_abfe_min.csv,
      4) computing Overestimation.

    Returns a normalized, sorted DataFrame matching the script schema.
    """
    exp_csv = expected_root / "exp_vs_abfe_min.csv"
    assert exp_csv.exists(), f"Missing synthesized exp/ABFE CSV: {exp_csv}"
    exp_df = pd.read_csv(exp_csv)
    need_cols = {"Lig_Name", "Vanilla", "Experimental_G", "MBAR"}
    assert need_cols.issubset(exp_df.columns), f"{exp_csv} missing columns {need_cols - set(exp_df.columns)}"

    rows = []
    # iterate over all compounds present
    for cmp_dir in sorted(p for p in expected_root.iterdir() if p.is_dir()):
        cmp_name = cmp_dir.name
        feat_csv = cmp_dir / "WaterOccupancy_features.csv"
        if not feat_csv.exists():
            continue
        features = pd.read_csv(feat_csv)
        assert {"ensemble", "occupancy"}.issubset(features.columns), f"{feat_csv} missing required columns."

        # (1) recompute avg
        avg_df = _recompute_avg_from_features(features)

        # decide window: use saved if present and we want to mirror script exactly
        saved_corr = expected_root / "water_occupancy_vs_delG.csv"
        saved_window = None
        if use_saved_window_if_present and saved_corr.exists():
            try:
                saved_df = pd.read_csv(saved_corr)
                if "window_id" in saved_df.columns and not saved_df.empty:
                    saved_window = str(saved_df["window_id"].iloc[0])
            except Exception:
                saved_window = None

        window_id = None
        if saved_window and (saved_window in avg_df["id"].astype(str).tolist()):
            window_id = saved_window
        else:
            window_id = _choose_window_id(avg_df, preferred=preferred_window)

        use = avg_df[avg_df["id"] == window_id].copy()
        for _, r in use.iterrows():
            vanilla = "van" + str(r["van"])
            match = exp_df[
                (exp_df["Lig_Name"] == int(cmp_name)) &
                (exp_df["Vanilla"]  == vanilla)
            ]
            if match.empty:
                # If synthetic table is correct, this shouldn't happen.
                continue
            rows.append({
                "compound":           str(cmp_name),
                "window_id":          window_id,
                "occupancy_selected": float(r["occupancy"]),
                "van":                str(r["van"]),
                "exp_delG":           float(match["Experimental_G"].iloc[0]),
                "pred_delG":          float(match["MBAR"].iloc[0]),
            })

    corr = pd.DataFrame(rows)
    assert not corr.empty, "Recomputed correlation is empty; check inputs."
    corr["Overestimation"] = corr["pred_delG"] - corr["exp_delG"]
    corr = _normalize_corr(corr)
    corr = _round_numeric(corr)
    corr = _sort_by(corr, keys=("compound", "van"))
    return corr


# --------------------- File discovery helpers ---------------------
def _pick_cmp_with_files(expected_root: Path) -> str:
    """Pick the first compound directory that has both required CSVs."""
    for cmp_dir in sorted(expected_root.iterdir()):
        if not cmp_dir.is_dir():
            continue
        if (cmp_dir / "WaterOccupancy_features.csv").exists() and (cmp_dir / "avg_water_occupancy.csv").exists():
            return cmp_dir.name
    raise FileNotFoundError(
        f"No compound folder under {expected_root} contains both "
        "WaterOccupancy_features.csv and avg_water_occupancy.csv"
    )


# --------------------- Tests ---------------------
def test_avg_and_correlation_csvs(tmp_path):
    # FEPA/ as repo root (this test file lives under FEPA/tests/...)
    repo_root = Path(__file__).resolve().parents[2]
    expected_root = repo_root / "tests" / "test_data" / "2_expected"
    assert expected_root.exists(), f"Missing expected root: {expected_root}"

    # ---------- A) aggregation CSV check (per compound) ----------
    cmp_name = _pick_cmp_with_files(expected_root)
    cmp_dir  = expected_root / cmp_name

    feat_csv = cmp_dir / "WaterOccupancy_features.csv"
    agg_csv  = cmp_dir / "avg_water_occupancy.csv"
    assert feat_csv.exists(), f"Missing features CSV: {feat_csv}"
    assert agg_csv.exists(),  f"Missing saved aggregation CSV: {agg_csv}"

    feat_df     = pd.read_csv(feat_csv)
    recomputed  = _recompute_avg_from_features(feat_df)
    tmp_agg     = tmp_path / f"{cmp_name}_avg_water_occupancy.recomputed.csv"
    recomputed.to_csv(tmp_agg, index=False)

    saved = pd.read_csv(agg_csv)
    for col in ["id", "van", "occupancy"]:
        assert col in saved.columns, f"Saved aggregation missing column '{col}'"

    saved_std      = _round_numeric(_normalize_agg(saved[["id", "van", "occupancy"]]))
    recomputed_std = _round_numeric(_normalize_agg(recomputed[["id", "van", "occupancy"]]))

    saved_std      = _sort_by(saved_std, keys=("id", "van"))
    recomputed_std = _sort_by(recomputed_std, keys=("id", "van"))

    # align on intersection of (id, van)
    keys = ["id", "van"]
    merged_keys = recomputed_std[keys].merge(saved_std[keys].drop_duplicates(), on=keys, how="inner")
    assert not merged_keys.empty, "No common (id, van) pairs between recomputed and saved aggregations."

    rc = recomputed_std.merge(merged_keys, on=keys, how="inner")
    sv = saved_std.merge(merged_keys, on=keys, how="inner")
    rc = _sort_by(rc, keys=("id", "van"))
    sv = _sort_by(sv, keys=("id", "van"))

    assert len(rc) == len(sv), "Row counts differ after alignment on (id, van)"

    pdt.assert_frame_equal(
        rc.reset_index(drop=True),
        sv.reset_index(drop=True),
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )

    # ---------- B) correlation CSV check (global) ----------
    corr_csv = expected_root / "water_occupancy_vs_delG.csv"
    assert corr_csv.exists(), f"Missing saved correlation CSV: {corr_csv}"
    corr_saved = pd.read_csv(corr_csv)
    need_cols = {"compound","window_id","occupancy_selected","van","exp_delG","pred_delG","Overestimation"}
    assert need_cols.issubset(corr_saved.columns), f"Saved correlation missing columns {need_cols - set(corr_saved.columns)}"
    assert not corr_saved.empty, "Saved correlation CSV is empty."

    # Recompute correlation from goldens + exp_vs_abfe_min.csv
    # Try to mirror the script’s selected window (if present), else fallback with preference for your tiny set
    corr_recomputed = _recompute_correlation_from_features_and_exp(
        expected_root,
        preferred_window="coul.01",              # sensible default for truncated data
        use_saved_window_if_present=True,        # mirror the script's window if saved
    )

    # Write recomputed to tmp for debugging
    (tmp_path / "water_occupancy_vs_delG.recomputed.csv").write_text(corr_recomputed.to_csv(index=False))

    # Normalize both and compare on intersection of keys (compound, van, window_id)
    corr_saved_n      = _round_numeric(_normalize_corr(corr_saved))
    corr_recomputed_n = _round_numeric(_normalize_corr(corr_recomputed))

    keys_corr = ["compound", "van", "window_id"]
    mk = corr_recomputed_n[keys_corr].merge(corr_saved_n[keys_corr].drop_duplicates(), on=keys_corr, how="inner")
    assert not mk.empty, "No common (compound, van, window_id) keys between recomputed and saved correlation."

    rc_corr = corr_recomputed_n.merge(mk, on=keys_corr, how="inner")
    sv_corr = corr_saved_n.merge(mk, on=keys_corr, how="inner")
    rc_corr = _sort_by(rc_corr, keys=("compound","van","window_id"))
    sv_corr = _sort_by(sv_corr, keys=("compound","van","window_id"))

    # sanity: Overestimation = pred - exp
    assert np.allclose(
        sv_corr["Overestimation"].to_numpy(),
        (sv_corr["pred_delG"] - sv_corr["exp_delG"]).to_numpy(),
        rtol=1e-12, atol=1e-12
    )

    pdt.assert_frame_equal(
        rc_corr.reset_index(drop=True)[list(need_cols)],
        sv_corr.reset_index(drop=True)[list(need_cols)],
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )
