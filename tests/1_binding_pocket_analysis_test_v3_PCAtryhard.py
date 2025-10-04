"""
1_binding_pocket_analysis_test using a minimal dataset.

Runs a tiny slice of the FEPA pipeline and verifies:
  1) required artifacts exist, and
  2) generated CSVs match reference CSVs (numeric tol; PCA sign-safe).
"""

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*importlib.*")

import os
from pathlib import Path
import builtins
from typing import Literal

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

builtins.Literal = Literal

from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.flows import binding_pocket_analysis_workflow

DECIMALS = 6
RTOL = 1e-6
ATOL = 1e-8

def _get_pc_cols(df: pd.DataFrame, prefix="PC") -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def _inner_join_on_keys(left: pd.DataFrame, right: pd.DataFrame,
                        keys=("ensemble", "timestep", "frame")) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [k for k in keys if k in left.columns and k in right.columns]
    if not keys:
        n = min(len(left), len(right))
        return left.iloc[:n].reset_index(drop=True), right.iloc[:n].reset_index(drop=True)
    # keep only common rows (same keys)
    common = left[keys].merge(right[keys], on=keys, how="inner").drop_duplicates()
    L = left.merge(common, on=keys, how="inner")
    R = right.merge(common, on=keys, how="inner")
    # sort for stable alignment
    sort_keys = [k for k in ("ensemble", "timestep", "frame") if k in keys]
    if sort_keys:
        L = L.sort_values(sort_keys).reset_index(drop=True)
        R = R.sort_values(sort_keys).reset_index(drop=True)
    return L, R

def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def _pairwise_dist_upper(X: np.ndarray) -> np.ndarray:
    """Euclidean distances, upper triangle flattened."""
    # D^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    norms = (X * X).sum(axis=1, keepdims=True)
    D2 = norms + norms.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, dtype=float)
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]

def _calibrated_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fit y ≈ a*x + b and return RMSE after calibration."""
    x = y_pred
    y = y_true
    x_mean = x.mean()
    y_mean = y.mean()
    # slope a = Cov(x,y)/Var(x); intercept b = y_mean - a*x_mean
    varx = ((x - x_mean) ** 2).mean()
    if varx == 0:
        a = 0.0
        b = y_mean
    else:
        a = ((x - x_mean) * (y - y_mean)).mean() / varx
        b = y_mean - a * x_mean
    y_hat = a * x + b
    return float(np.sqrt(((y_hat - y) ** 2).mean()))

def _round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        df[num_cols] = df[num_cols].round(DECIMALS)
    return df

def _sort_by(df: pd.DataFrame, keys=("ensemble", "timestep", "frame")) -> pd.DataFrame:
    use = [k for k in keys if k in df.columns]
    return df.sort_values(use).reset_index(drop=True) if use else df

def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate files exist: {paths!r}")

def _inner_join_on_keys(left: pd.DataFrame, right: pd.DataFrame,
                        keys=("ensemble", "timestep", "frame")) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [k for k in keys if k in left.columns and k in right.columns]
    if not keys:
        # Fall back to positional if no keys; keep same length
        n = min(len(left), len(right))
        return left.iloc[:n].reset_index(drop=True), right.iloc[:n].reset_index(drop=True)
    merged = left.merge(right[keys].drop_duplicates(), on=keys, how="inner")
    # Reindex both by the merged keys to ensure aligned order
    left2  = left.merge(merged[keys].drop_duplicates(), on=keys, how="inner")
    right2 = right.merge(merged[keys].drop_duplicates(), on=keys, how="inner")
    left2  = _sort_by(left2, keys)
    right2 = _sort_by(right2, keys)
    return left2.reset_index(drop=True), right2.reset_index(drop=True)

def _get_pc_cols(df: pd.DataFrame, prefix="PC") -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def _zscore_cols(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd

def _orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Find R (orthogonal) minimizing ||A R - B||_F.
    Returns (R, frobenius_norm).
    A, B are (n x d), already centered/scaled as desired.
    """
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    resid = np.linalg.norm(A @ R - B, ord="fro")
    return R, resid

def _assert_pca_equivalent(actual_df: pd.DataFrame, expected_df: pd.DataFrame,
                           keys=("ensemble", "timestep", "frame"),
                           pc_prefix="PC", rmsd_tol=1e-6):
    """Compare PCA scores up to translation/rotation/scale using Procrustes."""
    # Align rows
    A, E = _inner_join_on_keys(actual_df, expected_df, keys=keys)
    # Columns to compare
    pcs = [c for c in _get_pc_cols(A, pc_prefix) if c in E.columns]
    assert pcs, "No common PC columns to compare."
    # Extract, z-score each matrix column-wise
    XA = A[pcs].to_numpy(dtype=float)
    XE = E[pcs].to_numpy(dtype=float)
    # Equalize lengths if needed
    n = min(len(XA), len(XE))
    XA, XE = XA[:n], XE[:n]
    ZA, _, _ = _zscore_cols(XA)
    ZE, _, _ = _zscore_cols(XE)
    # Procrustes (orthogonal) alignment
    R, resid = _orthogonal_procrustes(ZE, ZA)  # map expected -> actual
    ZER = ZE @ R
    # RMSD per element after alignment
    rmsd = np.sqrt(np.mean((ZER - ZA) ** 2))
    assert rmsd <= rmsd_tol, f"PCA projections differ beyond rotation/shift tolerances (rmsd={rmsd:.3e} > {rmsd_tol})"

@pytest.mark.integration
def test_minimal_bp_analysis(tmp_path):
    # --- setup & run pipeline (unchanged from your clean test) ---
    test_root = Path(__file__).resolve().parents[1]
    config_path = test_root / "tests" / "test_config" / "config.json"
    assert config_path.exists(), f"Missing configuration file: {config_path}"
    config = load_config(str(config_path))

    cmp_name = config["compounds"][0]
    analysis_output_dir = tmp_path / "a1_binding_pocket_analysis_apo_holo_reus" / "wdir"
    cmp_output_dir = analysis_output_dir / cmp_name
    os.makedirs(cmp_output_dir, exist_ok=True)

    van_list = [1]
    leg_window_list = [f"coul.{i:02}" for i in range(2)]

    path_dict = load_abfe_paths_for_compound(
        config,
        cmp_name,
        van_list=van_list,
        leg_window_list=leg_window_list,
        bp_selection_string="name CA and resid " + config["pocket_residues_string"],
        apo=True,
    )

    for key in list(path_dict.keys()):
        if key.startswith("apo_") and key != "apo_1":
            del path_dict[key]

    ensemble_handler = EnsembleHandler(path_dict)
    workflow = binding_pocket_analysis_workflow(cmp_name, ensemble_handler)
    workflow.check_consistency()
    workflow.featurize(str(cmp_output_dir))
    workflow.reduce_dimensions(str(cmp_output_dir))
    workflow.cluster_pca(n_clusters=2, n_components=2)
    workflow.plot_pca_cluster(str(cmp_output_dir))
    workflow.load_results(str(cmp_output_dir))
    workflow.plot_holo_pca(str(cmp_output_dir))
    workflow.plot_apo_holo_pca(str(cmp_output_dir), apo_list=["apo_1"])
    workflow.plot_apo_abfe_pca(str(cmp_output_dir))

    assert (cmp_output_dir / "pca_projection_df.csv").exists(), "Missing PCA projection file"
    assert list(cmp_output_dir.glob("*.png")), "No plots were generated by the analysis"

    # --- strict features CSV compare (as before) ---
    expected_dir = test_root / "tests" / "test_data" / "1_expected"
    exp_feat_path = _first_existing(expected_dir / "SelfDistance_features.csv")
    act_feat_path = _first_existing(cmp_output_dir / "SelfDistance_features.csv")

    act_feat = _round_numeric(pd.read_csv(act_feat_path))
    exp_feat = _round_numeric(pd.read_csv(exp_feat_path))
    if "ensemble" in act_feat.columns and "ensemble" in exp_feat.columns:
        keep = set(act_feat["ensemble"].unique())
        exp_feat = exp_feat[exp_feat["ensemble"].isin(keep)].copy()

    act_feat = _sort_by(act_feat)
    exp_feat = _sort_by(exp_feat)

    assert list(act_feat.columns) == list(exp_feat.columns), "Feature CSV columns differ"
    assert len(act_feat) == len(exp_feat), "Feature CSV row counts differ"

    pdt.assert_frame_equal(
        act_feat.reset_index(drop=True),
        exp_feat.reset_index(drop=True),
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=False,
    )

    # --- PCA CSV compare: geometry only (rotation/translation/scale invariant) ---
    act_pca = pd.read_csv(cmp_output_dir / "pca_projection_df.csv")
    exp_pca = pd.read_csv(expected_dir / "pca_projection_df.csv")

    # Align rows on common keys
    A, E = _inner_join_on_keys(act_pca, exp_pca, keys=("ensemble", "timestep", "frame"))

    # Select common PC columns
    pcs = [c for c in _get_pc_cols(A) if c in E.columns]
    assert pcs, "No common PC columns to compare."
    XA = _zscore_cols(A[pcs].to_numpy(dtype=float))
    XE = _zscore_cols(E[pcs].to_numpy(dtype=float))

    # Equalize length (paranoia)
    n = min(len(XA), len(XE))
    XA, XE = XA[:n], XE[:n]

    # Pairwise distance geometry
    dA = _pairwise_dist_upper(XA)
    dE = _pairwise_dist_upper(XE)

    # Pearson correlation (no scipy needed)
    corr = np.corrcoef(dA, dE)[0, 1]
    rmse = _calibrated_rmse(dA, dE)
    nrmse = rmse / (dA.mean() if dA.mean() != 0 else 1.0)

    # Tolerances: lenient enough to ignore tiny numeric drift, strict enough to catch regressions
    assert corr >= 0.999, f"PCA geometry correlation too low: {corr:.6f} < 0.999"
    assert nrmse <= 5e-3, f"PCA geometry NRMSE too high: {nrmse:.3e} > 5e-3"

