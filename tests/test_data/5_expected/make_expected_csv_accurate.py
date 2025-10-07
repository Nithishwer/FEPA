#!/usr/bin/env python3
from __future__ import annotations

import os
import logging
from pathlib import Path
import tempfile
import pandas as pd

from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.core.featurizers import SideChainTorsionsFeaturizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
LOG = logging.getLogger("make_expected_sct_csv")


def _repo_root() -> Path:
    """Find repo root by walking up until we see pyproject.toml or .git."""
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here.parents[2]


def _filter_existing_nonempty_pairs(path_dict: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    kept: dict[str, dict[str, str]] = {}
    for ens, d in path_dict.items():
        tpr = Path(d.get("tpr", ""))
        xtc = Path(d.get("xtc", ""))
        try:
            if tpr.exists() and xtc.exists() and tpr.stat().st_size > 0 and xtc.stat().st_size > 0:
                kept[ens] = d
            else:
                LOG.info("Skip %s (missing/empty): %s | %s", ens, tpr, xtc)
        except OSError:
            LOG.info("Skip %s (OS error): %s | %s", ens, tpr, xtc)
    return kept


def _detect_leg_with_data(data_base: Path) -> str | None:
    """Return 'coul.00' or 'coul.01' if a usable (tpr, xtc) exists under prod/; else None."""
    for leg in ("coul.00", "coul.01"):
        prod = data_base / leg / "prod"
        tpr = (prod / "prod.tpr") if (prod / "prod.tpr").exists() else (prod / "topol.tpr")
        xtc = (prod / "prod.xtc") if (prod / "prod.xtc").exists() else (prod / "traj_comp.xtc")
        if tpr.exists() and xtc.exists() and tpr.stat().st_size > 0 and xtc.stat().st_size > 0:
            return leg
    return None


def main() -> None:
    # Keep any FEPA side-effects away from the repo
    with tempfile.TemporaryDirectory(prefix="fepa_wdir_") as tmpw:
        os.environ["FEPA_WDIR"] = tmpw

        repo = _repo_root()
        cfg_path = repo / "tests" / "test_config" / "config.json"
        assert cfg_path.exists(), f"Missing config: {cfg_path}"

        config = load_config(str(cfg_path))
        compounds = list(map(str, config.get("compounds", [])))
        assert compounds, "Config has no 'compounds'"
        cmp_name = compounds[0]

        data_base = repo / "tests" / "test_data" / cmp_name
        assert (data_base / "vanilla_rep_1" / "npt.gro").exists(), "missing vanilla_rep_1/npt.gro under tests data"

        leg = _detect_leg_with_data(data_base)
        if not leg:
            raise FileNotFoundError("Could not find a valid (tpr, xtc) under coul.00 or coul.01/prod in tests data")

        LOG.info("Compound: %s | Using leg: %s", cmp_name, leg)

        # Build FEPA path dict (same entrypoint as the original script)
        LOG.info("Building path_dict via FEPA loader…")
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp_name,
            van_list=[1],                                # match test constraints
            leg_window_list=[leg],                       # coul.00 or coul.01
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )

        # Make paths absolute so MDAnalysis can open them regardless of CWD
        fixed: dict[str, dict[str, str]] = {}
        for ens, d in path_dict.items():
            newd = dict(d)
            tpr = Path(newd["tpr"])
            xtc = Path(newd["xtc"])
            if not tpr.is_absolute():
                tpr = (repo / tpr).resolve()
            if not xtc.is_absolute():
                xtc = (repo / xtc).resolve()
            newd["tpr"] = str(tpr)
            newd["xtc"] = str(xtc)
            # ensure bp_selection_string exists (EnsembleHandler expects it)
            if not newd.get("bp_selection_string"):
                newd["bp_selection_string"] = "name CA and resid " + config["pocket_residues_string"]
            fixed[ens] = newd

        fixed = _filter_existing_nonempty_pairs(fixed)
        if not fixed:
            raise FileNotFoundError("After filtering, no usable (tpr, xtc) pairs remain in tests data")

        LOG.info("Creating universes…")
        eh = EnsembleHandler(fixed)
        eh.make_universes()

        LOG.info("Featurizing side-chain torsions…")
        sct = SideChainTorsionsFeaturizer(ensemble_handler=eh)
        sct.featurize()

        # Write EXACT output where your test expects it
        out_dir = repo / "tests" / "test_data" / "5_expected" / cmp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "SideChainTorsions_features.csv"

        LOG.info("Saving features to %s", out_csv)
        sct.save_features(str(out_dir), overwrite=True)

        # sanity
        assert out_csv.exists() and out_csv.stat().st_size > 0, f"Output CSV missing or empty: {out_csv}"
        df = pd.read_csv(out_csv)
        assert "ensemble" in df.columns and "timestep" in df.columns, "CSV missing core columns"
        assert any(c.startswith("CHI") for c in df.columns), "CSV has no CHI* columns"
        LOG.info("Done. Wrote %d rows, %d columns.", len(df), len(df.columns))


if __name__ == "__main__":
    main()
