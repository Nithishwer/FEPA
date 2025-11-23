import logging
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt
import seaborn as sns

from fepa.utils.file_utils import load_config  # only FEPA import we need

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- helpers ----------------
def _abspath_templates(config: dict, repo_root: Path) -> dict:
    """Prefix repo_root to any relative templates in the test config."""
    keys = [
        "abfe_window_path_template",
        "vanilla_path_template",
        "vanilla_path_template_old",
        "apo_path_template",
    ]
    out = dict(config)
    for k in keys:
        if k in out:
            p = Path(out[k])
            if not p.is_absolute():
                out[k] = str((repo_root / p).resolve())
    return out

def _extract_window(series: pd.Series) -> pd.Series:
    # Find vdw.NN / rest.NN / coul.NN anywhere in the string
    return series.str.extract(r"(vdw\.\d+|rest\.\d+|coul\.\d+)", expand=False)

def _extract_van(series: pd.Series) -> pd.Series:
    # Extract vanilla index from 'van_1' or 'van-1' anywhere
    return series.str.extract(r"van[_-]?(\d+)", expand=False)

def _choose_window_id(avg_df: pd.DataFrame, preferred: str | None) -> str | None:
    """Pick a usable window id from avg_df['id'] (preferred -> max vdw -> max rest -> max coul -> first)."""
    if "id" not in avg_df.columns or avg_df.empty:
        return None
    ids = avg_df["id"].dropna().astype(str)
    if ids.empty:
        return None
    if preferred and (ids == preferred).any():
        return preferred
    parsed = ids.str.extract(r"^(vdw|rest|coul)\.(\d+)$")
    if not parsed.isna().all().all():
        tmp = pd.DataFrame({"id": ids})
        tmp["_fam"] = parsed[0]
        tmp["_idx"] = pd.to_numeric(parsed[1], errors="coerce")
        tmp = tmp.dropna(subset=["_fam", "_idx"])
        for fam in ("vdw", "rest", "coul"):
            sub = tmp[tmp["_fam"] == fam]
            if not sub.empty:
                max_idx = int(sub["_idx"].max())
                return f"{fam}.{max_idx:02d}"
    return ids.iloc[0]

def _ensure_exp_abfe_table(exp_path: Path,
                           needed_pairs: list[tuple[int, str]],
                           test_mode: bool,
                           default_exp: float = -6.50,
                           default_pred: float = -7.20) -> pd.DataFrame:
    """
    Load an exp/ABFE table from exp_path. If test_mode is True, synthesize any
    missing (Lig_Name, Vanilla) pairs with default values and write back.
    """
    if exp_path.exists():
        exp_df = pd.read_csv(exp_path)
    else:
        exp_df = pd.DataFrame(columns=["Lig_Name", "Vanilla", "Experimental_G", "MBAR"])

    # Normalize types
    if "Lig_Name" in exp_df.columns:
        exp_df["Lig_Name"] = pd.to_numeric(exp_df["Lig_Name"], errors="coerce").astype("Int64")
    if "Vanilla" in exp_df.columns:
        exp_df["Vanilla"] = exp_df["Vanilla"].astype(str)

    if test_mode:
        have = set(
            (int(ln) if pd.notna(ln) else None, v)
            for ln, v in zip(exp_df.get("Lig_Name", pd.Series(dtype="Int64")), exp_df.get("Vanilla", pd.Series(dtype=str)))
            if pd.notna(ln) and isinstance(v, str)
        )
        needed = set(needed_pairs)
        add = needed - have
        if add:
            synth = pd.DataFrame({
                "Lig_Name":       [lig for lig, _ in add],
                "Vanilla":        [van for _, van in add],
                "Experimental_G": [default_exp] * len(add),
                "MBAR":           [default_pred] * len(add),
            })
            exp_df = pd.concat([exp_df, synth], ignore_index=True)
            exp_df.to_csv(exp_path, index=False)
            logging.info("Synthesized %d exp/ABFE rows into %s (FEPA_TEST_MODE=1).", len(add), exp_path)

    # Final type cleanup
    exp_df["Lig_Name"] = pd.to_numeric(exp_df["Lig_Name"], errors="coerce").astype("Int64")
    exp_df["Vanilla"] = exp_df["Vanilla"].astype(str)
    return exp_df

# ---------------- main ----------------
def main():
    """Correlate water occupancy (selected window) vs ΔG using test goldens; force outputs in test mode."""
    # FEPA repo root & test config
    repo_root = Path(__file__).resolve().parents[3]  # FEPA/
    cfg_path  = repo_root / "tests" / "test_config" / "config.json"
    config    = load_config(str(cfg_path))
    config    = _abspath_templates(config, repo_root)

    # I/O anchored in test goldens dir
    analysis_output_dir = (repo_root / "tests" / "test_data" / "2_expected").resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Env knobs
    preferred_window = os.environ.get("FEPA_TARGET_WINDOW")  # e.g., 'coul.01' for truncated set
    test_mode        = os.environ.get("FEPA_TEST_MODE", "0") in ("1", "true", "True")

    # Determine exp/ABFE CSV source:
    #  - if FEPA_EXP_ABFE_CSV is provided, use it;
    #  - elif FEPA_TEST_MODE=1, use (and synthesize) analysis_output_dir/exp_vs_abfe_min.csv;
    #  - else fall back to the large shared CSV path.
    exp_env = os.environ.get("FEPA_EXP_ABFE_CSV")
    if exp_env:
        exp_csv_path = Path(exp_env)
    elif test_mode:
        exp_csv_path = analysis_output_dir / "exp_vs_abfe_min.csv"
    else:
        exp_csv_path = Path("/biggin/b211/reub0138/Projects/orexin/christopher_et_al_v1/exp_v_abfe_df_van_reps_all_hrex.csv")

    correlation_rows = []
    needed_pairs: list[tuple[int, str]] = []  # (Lig_Name, Vanilla)

    for cmp in config["compounds"]:
        logging.info("Analyzing compound %s ...", cmp)

        cmp_output_dir = analysis_output_dir / str(cmp)
        cmp_output_dir.mkdir(parents=True, exist_ok=True)

        feat_path = cmp_output_dir / "WaterOccupancy_features.csv"
        if not feat_path.exists():
            logging.warning("Missing %s — skipping compound %s", feat_path, cmp)
            continue

        features_df = pd.read_csv(feat_path)

        # Robust extraction of window + vanilla
        features_df["id"]  = _extract_window(features_df["ensemble"])
        features_df["van"] = _extract_van(features_df["ensemble"])
        base = features_df.dropna(subset=["id", "van"]).copy()
        if base.empty:
            logging.warning("No parseable rows in %s; skipping compound %s", feat_path, cmp)
            continue

        # Average occupancy by (window, van)
        avg_df = base.groupby(["id", "van"], as_index=False)["occupancy"].mean()
        avg_df.to_csv(cmp_output_dir / "avg_water_occupancy.csv", index=False)

        # Choose a window present in this compound’s data
        chosen = _choose_window_id(avg_df, preferred=preferred_window)
        if not chosen:
            logging.warning("No usable window id found for cmp=%s; skipping", cmp)
            continue
        logging.info("Using window id '%s' for cmp=%s", chosen, cmp)

        use_df = avg_df[avg_df["id"] == chosen].copy()
        if use_df.empty:
            continue

        # Collect pairs needed for exp/ABFE join
        for _, row in use_df.iterrows():
            vanilla = "van" + str(row["van"])
            needed_pairs.append((int(cmp), vanilla))
            correlation_rows.append({
                "compound":            str(cmp),
                "window_id":           chosen,
                "occupancy_selected":  float(row["occupancy"]),
                "van":                 str(row["van"]),
                "Lig_Name":            int(cmp),
                "Vanilla":             vanilla,
            })

    if not correlation_rows:
        logging.warning("No rows to correlate; nothing to write.")
        return

    # Ensure exp/ABFE table contains all needed pairs (synthesize if FEPA_TEST_MODE=1)
    exp_df = _ensure_exp_abfe_table(exp_csv_path, needed_pairs, test_mode=test_mode)

    # Join & finalize
    corr = pd.DataFrame(correlation_rows)
    corr = corr.merge(
        exp_df[["Lig_Name", "Vanilla", "Experimental_G", "MBAR"]],
        on=["Lig_Name", "Vanilla"],
        how="left",
    )

    missing = corr["Experimental_G"].isna() | corr["MBAR"].isna()
    if missing.any():
        nmiss = int(missing.sum())
        if test_mode:
            logging.warning("Unexpected: %d rows still missing exp/ABFE even in test mode.", nmiss)
        else:
            logging.warning("Dropping %d rows without exp/ABFE values.", nmiss)
            corr = corr[~missing].copy()

    if corr.empty:
        logging.warning("Correlation table empty after join; nothing to write/plot.")
        return

    corr = corr.rename(columns={"Experimental_G": "exp_delG", "MBAR": "pred_delG"})
    corr["Overestimation"] = corr["pred_delG"] - corr["exp_delG"]

    # Write CSV
    corr_csv = analysis_output_dir / "water_occupancy_vs_delG.csv"
    corr.sort_values(["compound", "van"]).to_csv(corr_csv, index=False)

    # Plots
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=corr, x="occupancy_selected", y="Overestimation", hue="van", palette="tab10", s=100)
    for _, r in corr.iterrows():
        plt.text(r["occupancy_selected"], r["Overestimation"], str(r["compound"]), fontsize=9, ha="right")
    plt.title("Water Occupancy vs Overestimation of Binding Free Energy")
    plt.xlabel("Water Occupancy (selected window)")
    plt.ylabel("Overestimation (kcal/mol)")
    plt.legend(title="Vanilla Runs", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(analysis_output_dir / "water_occupancy_vs_overestimation.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=corr, x="occupancy_selected", y="pred_delG", hue="van", palette="tab10", s=100)
    for _, r in corr.iterrows():
        plt.text(r["occupancy_selected"], r["pred_delG"], str(r["compound"]), fontsize=9, ha="right")
    plt.title("Water Occupancy vs pred_delG")
    plt.xlabel("Water Occupancy (selected window)")
    plt.ylabel("pred_delG (kcal/mol)")
    plt.legend(title="Vanilla Runs", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(analysis_output_dir / "water_occupancy_vs_pred_delG.png")

if __name__ == "__main__":
    main()
