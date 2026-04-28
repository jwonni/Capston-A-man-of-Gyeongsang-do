#!/usr/bin/env python3
"""Test script to generate feature matching visualizations"""

import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent / "path-to-gpx" / "src"))

# Run the main script with visualization enabled
import subprocess

result = subprocess.run([
    sys.executable,
    "path-to-gpx/src/05_loftr_match.py",
    "--anchors-json", "path-to-gpx/output/04.anchors/083_anchors.json",
    "--confidence", "0.2",
    "--magsac-thr", "3.0",
    "--output-dir", "path-to-gpx/output/05.loftr/",
    "--save-visualizations"
], cwd=str(Path(__file__).parent))

sys.exit(result.returncode)
