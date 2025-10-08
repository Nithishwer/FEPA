"""
End-to-end pytest for the FEPA side chain torsions featurization pipeline.

This test runs the actual FEPA pipeline without mocks and validates:
1. Pipeline runs successfully with minimal test data
2. All expected files are produced
3. Generated CSV matches golden truth CSV exactly

This test is completely self-contained and independent.
"""

import logging
import os
import tempfile
import shutil
from pathlib import Path
import re
import math

import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

# Import FEPA package components
from fepa.utils.file_utils import load_config
from fepa.core.ensemble_handler import EnsembleHandler
from fepa.utils.path_utils import (
    load_paths_for_compound,
    load_abfe_paths_for_compound,
    load_paths_for_apo,
)
from fepa.core.featurizers import SideChainTorsionsFeaturizer
from fepa.utils.dimred_utils import cluster_pca
from fepa.core.dim_reducers import PCADimReducer
from fepa.core.visualizers import (
    DimRedVisualizer,
    plot_eigenvalues,
    plot_pca_components,
    plot_entropy_heatmaps,
)
from fepa.utils.dimred_utils import (
    cluster_pca,
    get_ensemble_center,
    make_ensemble_center_df,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


class TestFEPAPipelineE2E:
    """End-to-end test of the FEPA pipeline."""
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test configuration file."""
        return Path(__file__).parent.parent / "test_config" / "config.json"
    
    @pytest.fixture
    def expected_csv_path(self):
        """Path to golden truth CSV file."""
        return Path(__file__).parent.parent / "test_data" / "5_expected" / "1" / "SideChainTorsions_features.csv"
    
    @pytest.fixture
    def output_dir(self, tmp_path):
        """Temporary output directory for test results."""
        output_dir = tmp_path / "test_output" / "1"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def test_generated_csv_structure(self, test_config_path, output_dir, expected_csv_path):
        """Test that generated CSV has correct structure."""
        if not test_config_path.exists():
            pytest.skip("Test config file not found")
        
        # Run the pipeline
        config = load_config(str(test_config_path))
        repo_root = Path(__file__).parent.parent.parent
        config = _abspath_templates(config, repo_root)
        
        cmp = "1"
        cmp_output_dir = output_dir
        cmp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline steps
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=[1],
            leg_window_list=[f"coul.{i:02}" for i in range(2)],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )
        
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()
        
        sct_featurizer = SideChainTorsionsFeaturizer(ensemble_handler=ensemble_handler)
        sct_featurizer.featurize()
        sct_featurizer.save_features(str(cmp_output_dir), overwrite=True)
        
        # Check generated CSV
        generated_csv_file = cmp_output_dir / "SideChainTorsions_features.csv"
        assert generated_csv_file.exists()
        
        generated_df = pd.read_csv(generated_csv_file)
        
        # Verify structure
        assert 'timestep' in generated_df.columns
        assert 'ensemble' in generated_df.columns
        
        # Check for CHI columns
        chi_columns = [col for col in generated_df.columns if 'CHI' in col]
        assert len(chi_columns) > 0, "No CHI columns found in generated CSV"
        
        # Check data types
        ts = generated_df['timestep']
        assert np.issubdtype(ts.dtype, np.number), f"timestep dtype must be numeric, got {ts.dtype}"
        # if it's float, allow only values that are close to integers
        if np.issubdtype(ts.dtype, np.floating):
            assert np.allclose(ts.values, np.round(ts.values), atol=1e-9), "timestep values should be integer-like"
        # ensemble should be string-like
        assert (generated_df['ensemble'].dtype == 'object') or pd.api.types.is_string_dtype(generated_df['ensemble'])
        
        # Check for reasonable data ranges
        assert generated_df['timestep'].min() >= 0
        assert len(generated_df['ensemble'].unique()) > 0
        
        # Verify ensembles follow expected naming
        ensembles = generated_df['ensemble'].unique()
        for ensemble in ensembles:
            assert isinstance(ensemble, str)
            assert len(ensemble) > 0
    
    def test_csv_matches_golden_truth(self, test_config_path, output_dir, expected_csv_path):
        """Test that generated CSV matches golden truth CSV exactly."""
        if not test_config_path.exists():
            pytest.skip("Test config file not found")
        
        if not expected_csv_path.exists():
            pytest.skip("Golden truth CSV file not found")
        
        # Run the pipeline
        config = load_config(str(test_config_path))
        repo_root = Path(__file__).parent.parent.parent
        config = _abspath_templates(config, repo_root)
        
        cmp = "1"
        cmp_output_dir = output_dir
        cmp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline steps
        path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=[1],
            leg_window_list=[f"coul.{i:02}" for i in range(2)],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )
        
        ensemble_handler = EnsembleHandler(path_dict)
        ensemble_handler.make_universes()
        
        sct_featurizer = SideChainTorsionsFeaturizer(ensemble_handler=ensemble_handler)
        sct_featurizer.featurize()
        sct_featurizer.save_features(str(cmp_output_dir), overwrite=True)
        
        # Load both CSVs
        generated_csv_file = cmp_output_dir / "SideChainTorsions_features.csv"
        assert generated_csv_file.exists()
        
        generated_df = pd.read_csv(generated_csv_file)
        expected_df = pd.read_csv(expected_csv_path)
        
        # Compare shapes
        assert generated_df.shape == expected_df.shape, f"Shape mismatch: {generated_df.shape} vs {expected_df.shape}"
        
        # Compare columns
        assert list(generated_df.columns) == list(expected_df.columns), "Column mismatch"
        
        # Compare data exactly (values must be identical)
        for col in generated_df.columns:
            if generated_df[col].dtype in ['float64', 'float32']:
                # For float columns, check exact equality (no tolerance)
                assert np.array_equal(generated_df[col].values, expected_df[col].values, equal_nan=True), \
                    f"Float column {col} values don't match exactly"
            else:
                # For non-float columns, check exact equality
                pd.testing.assert_series_equal(generated_df[col], expected_df[col], check_names=False), \
                    f"Non-float column {col} values don't match exactly"
        
        logging.info("Generated CSV matches golden truth CSV exactly")
    
    def test_main_pipeline_workflow(self, test_config_path, tmp_path):
        """Test the complete main pipeline workflow."""
        if not test_config_path.exists():
            pytest.skip("Test config file not found")
        
        # Create output directory structure
        analysis_output_dir = tmp_path / "tests" / "test_data" / "5_expected"
        analysis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        config = load_config(str(test_config_path))
        repo_root = Path(__file__).parent.parent.parent
        config = _abspath_templates(config, repo_root)
        
        # Process each compound (should be just "1")
        for cmp in config["compounds"]:
            logging.info(f"Analyzing compound {cmp}")
            
            # Create output directory
            cmp_output_dir = analysis_output_dir / cmp
            cmp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare paths
            logging.info(f"Loading paths for compound {cmp}")
            path_dict = load_abfe_paths_for_compound(
                config,
                cmp,
                van_list=[1],
                leg_window_list=[f"coul.{i:02}" for i in range(2)],
                bp_selection_string="name CA and resid " + config["pocket_residues_string"],
                apo=True,
            )
            
            # Load trajectories
            logging.info(f"Loading trajectories for compound {cmp}")
            ensemble_handler = EnsembleHandler(path_dict)
            ensemble_handler.make_universes()
            
            # Featurize
            logging.info("Featurizing side chain torsions")
            sct_featurizer = SideChainTorsionsFeaturizer(ensemble_handler=ensemble_handler)
            sct_featurizer.featurize()
            
            # Save features
            logging.info(f"Saving features for compound {cmp}")
            sct_featurizer.save_features(str(cmp_output_dir), overwrite=True)
            
            # Verify output was created
            csv_file = cmp_output_dir / "SideChainTorsions_features.csv"
            assert csv_file.exists(), f"CSV file not created for compound {cmp}"
            assert csv_file.stat().st_size > 0, f"CSV file is empty for compound {cmp}"
            
            logging.info(f"Pipeline completed successfully for compound {cmp}")
        
        # Verify output directory structure
        assert analysis_output_dir.exists()
        
        # Check if compound directory was created
        compound_dir = analysis_output_dir / "1"
        if compound_dir.exists():
            csv_file = compound_dir / "SideChainTorsions_features.csv"
            if csv_file.exists():
                assert csv_file.stat().st_size > 0, "Generated CSV file is empty"
                logging.info(f"Main pipeline produced CSV file: {csv_file}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])