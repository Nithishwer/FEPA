#!/usr/bin/env python3
"""
Simple runner for the end-to-end FEPA pipeline test.
"""

import sys
import subprocess
from pathlib import Path

def run_e2e_test():
    """Run the end-to-end test."""
    test_file = Path(__file__).parent / "test_sc_torsions_e2e.py"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False
    
    print(f"Running end-to-end FEPA pipeline test: {test_file}")
    
    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("=" * 80)
        print("STDOUT:")
        print(result.stdout)
        print("=" * 80)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            print("=" * 80)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = run_e2e_test()
    if success:
        print("✅ End-to-end test passed!")
        sys.exit(0)
    else:
        print("❌ End-to-end test failed!")
        sys.exit(1)
