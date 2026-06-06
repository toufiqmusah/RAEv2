#!/bin/bash
# ============================================================
# MED-RAEv2 Setup Script
# Installs dependencies and clones FRD-Score repo.
# ============================================================
set -euo pipefail

echo "=== Installing MED-RAEv2 dependencies ==="

# Install core dependencies
pip install open_clip_torch scikit-learn matplotlib

# Install FRD-Score for medical FID
if [ ! -d "frd-score" ]; then
    echo "=== Installing FRD-Score ==="
    git clone https://github.com/RichardObi/frd-score.git
    cd frd-score
    pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
    pip install -e ".[dev]"
    cd ..
    echo "FRD-Score installed."
else
    echo "FRD-Score already exists, skipping."
fi

echo "=== Setup complete ==="
