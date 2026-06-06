#!/bin/bash
# ============================================================
# MED-RAEv2: Stage 2 Training Launcher
#
# Usage:
#   ./scripts/run_stage2_med.sh dit-xl-medsiglip-l-k7       # DiT-XL + MedSigLIP MLS K7
#   ./scripts/run_stage2_med.sh dit-xl-medsiglip-l-k1       # DiT-XL + MedSigLIP K1
#   ./scripts/run_stage2_med.sh dit-xl-biomedclip-b-k4      # DiT-XL + BiomedCLIP MLS K4
#   ./scripts/run_stage2_med.sh dit-xl-biomedclip-b-k1      # DiT-XL + BiomedCLIP K1
#   ./scripts/run_stage2_med.sh dit-xl-dinov3-b-k1          # DiT-XL + DINOv3-B K1
#   ./scripts/run_stage2_med.sh dit-xl-dinov3-b-k4          # DiT-XL + DINOv3-B MLS K4
#   ./scripts/run_stage2_med.sh dit-xl-dinov3-l-k1          # DiT-XL + DINOv3-L K1
#   ./scripts/run_stage2_med.sh dit-xl-dinov3-l-k7          # DiT-XL + DINOv3-L MLS K7
#   ./scripts/run_stage2_med.sh dit-xl-medvae-4_3_2d        # DiT-XL + MedVAE 4x
#   ./scripts/run_stage2_med.sh dit-xl-medvae-8_4_2d        # DiT-XL + MedVAE 8x
# ============================================================
set -euo pipefail

RUN_NAME="${1:-dit-xl-medsiglip-l-k7}"
CONFIG="configs/stage2/training/medical/${RUN_NAME}.yaml"
RESULTS_DIR="results/medraev2/stage2/${RUN_NAME}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    echo "Available configs:"
    ls configs/stage2/training/medical/*.yaml
    exit 1
fi

echo "=== Stage 2 Training: ${RUN_NAME} ==="
echo "Config: ${CONFIG}"
echo "Results: ${RESULTS_DIR}"

# Multi-GPU (default)
torchrun --standalone --nproc_per_node=8 \
    src/train.py \
    --config "$CONFIG" \
    --results-dir "$RESULTS_DIR" \
    --precision bf16

# For single GPU add --wandb flag:
# torchrun --standalone --nproc_per_node=1 \
#     src/train.py \
#     --config "$CONFIG" \
#     --results-dir "$RESULTS_DIR" \
#     --precision bf16 \
#     --wandb
