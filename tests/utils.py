import numpy as np
import pandas as pd
from pathlib import Path
import pandas.testing as pdt

# === numeric comparison tolerances (module-level defaults)
DECIMALS = 6
RTOL = 1e-6
ATOL = 1e-8


# === small helpers for deterministic comparisons
def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Round all numeric columns in a DataFrame to a fixed number of decimals."""
    df = df.copy()
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].round(DECIMALS)
    return df


def first_existing(*paths: Path) -> Path:
    """Return the first existing path from a list of candidates."""
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate files exist: {paths!r}")


def sort_by(df: pd.DataFrame, keys=("ensemble", "timestep", "frame")) -> pd.DataFrame:
    """Sort a DataFrame by available keys, resetting the index."""
    use = [k for k in keys if k in df.columns]
    return df.sort_values(use).reset_index(drop=True) if use else df


def align_pc_signs(actual: pd.DataFrame, expected: pd.DataFrame, pc_prefix: str = "PC") -> pd.DataFrame:
    """Flip PC signs in 'expected' to match 'actual' for consistent orientation."""
    exp = expected.copy()
    pc_cols = [c for c in exp.columns if c.startswith(pc_prefix) and c in actual.columns]
    for c in pc_cols:
        a = actual[c].to_numpy()
        b = exp[c].to_numpy()
        if np.dot(a, b) < 0:
            exp[c] = -exp[c]
    return exp


def check_csv_equality(
    actual_path: Path,
    expected_path: Path,
    *,
    sort_keys=("ensemble", "timestep", "frame"),
    pc_prefix: str | None = None,
    decimals: int = DECIMALS,
    rtol: float = RTOL,
    atol: float = ATOL,
    label: str = "CSV file",
):
    """Assert equality between two CSVs within numeric tolerances and schema consistency."""
    # --- Load and normalize both CSVs
    act = sort_by(round_numeric(pd.read_csv(actual_path)), keys=sort_keys)
    exp = sort_by(round_numeric(pd.read_csv(expected_path)), keys=sort_keys)

    # Align PCs if applicable
    if pc_prefix:
        exp = align_pc_signs(act, exp, pc_prefix=pc_prefix)

    # --- Schema and shape checks
    assert list(act.columns) == list(exp.columns), (
        f"{label}: column mismatch.\n"
        f"Actual columns:   {list(act.columns)}\n"
        f"Expected columns: {list(exp.columns)}"
    )
    assert len(act) == len(exp), (
        f"{label}: row count mismatch "
        f"(actual={len(act)}, expected={len(exp)})."
    )

    # --- Numeric comparison
    try:
        pdt.assert_frame_equal(
            act.reset_index(drop=True),
            exp.reset_index(drop=True),
            check_exact=False,
            rtol=rtol,
            atol=atol,
            check_dtype=False,
        )
    except AssertionError as e:
        raise AssertionError(f"{label}: numeric values differ beyond tolerance.\n{e}") from e
