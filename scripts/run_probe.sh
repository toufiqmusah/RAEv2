#!/bin/bash
# ============================================================
# MED-RAEv2: Phase 0 Probing Suite (LP, LDS, CKA)
#
# Usage:
#   ./scripts/run_probe.sh <data_dir> [encoders] [output_dir]
#
# Examples:
#   ./scripts/run_probe.sh ../BiomedParseDataRAE/train
#   ./scripts/run_probe.sh ../BiomedParseDataRAE/train "medsiglip-vit-l,biomedclip-vit-b"
# ============================================================
set -euo pipefail

DATA_DIR="${1:?Error: data_dir required}"
# Default: all encoder variants for LP+LDS probing
# Includes: MedSigLIP-L (K=1, K=7), BiomedCLIP-B (K=1, K=4),
#           DINOv2-B (K=1, K=4), DINOv3-B (K=1, K=4), DINOv3-L (K=1),
#           MedVAE (4x3, 8x4)
# All 12 variants at consistent ~33% MLS (K=4 for 12-layer, K=7 for 24-27 layer)
ENCODERS="${2:-medsiglip-vit-l,medsiglipmls-vit-l[K=7],biomedclip-vit-b,biomedclipmls-vit-b[K=4],dinov2-vit-b,dinov2mls-vit-b[layers=8.9.10.11],dinov3-vit-b16,dinov3mls-vit-b16[layers=8.9.10.11],dinov3-vit-l16,dinov3mls-vit-l16[layers=11.13.15.17.19.21.23],medvae-cnn-4_3_2d,medvae-cnn-8_4_2d}"
OUTPUT="${3:-results/medraev2/probing}"

echo "=== Phase 0: Probing Suite ==="
echo "Data:    ${DATA_DIR}"
echo "Probing: ${ENCODERS}"
echo "Output:  ${OUTPUT}"

python scripts/probing/run_probe_suite.py \
    --data-dir "$DATA_DIR" \
    --encoders "$ENCODERS" \
    --output "$OUTPUT" \
    --batch-size 32 \
    --lds-samples 1000
