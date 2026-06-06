#!/bin/bash
# ============================================================
# MED-RAEv2: Stage 1 Training Launcher
#
# Usage:
#   ./scripts/run_stage1_med.sh medsiglip-l-k1         # MedSigLIP K=1
#   ./scripts/run_stage1_med.sh medsiglip-l-k7         # MedSigLIP K=7
#   ./scripts/run_stage1_med.sh biomedclip-b-k1        # BiomedCLIP K=1
#   ./scripts/run_stage1_med.sh biomedclip-b-k4        # BiomedCLIP K=4
#   ./scripts/run_stage1_med.sh dinov3-b-k1            # DINOv3-B K=1
#   ./scripts/run_stage1_med.sh dinov3-b-k4            # DINOv3-B MLS K=4
#   ./scripts/run_stage1_med.sh dinov3-l-k1            # DINOv3-L K=1
#   ./scripts/run_stage1_med.sh dinov3-l-k7            # DINOv3-L MLS K=7
#   ./scripts/run_stage1_med.sh medvae-4_3_2d          # MedVAE 4x (3-ch, 2D)
#   ./scripts/run_stage1_med.sh medvae-8_4_2d          # MedVAE 8x (3-ch, 2D)
# ============================================================
set -euo pipefail

RUN_NAME="${1:-medsiglip-l-k1}"
CONFIG="configs/stage1/training/medical/${RUN_NAME}.yaml"
RESULTS_DIR="results/medraev2/stage1/${RUN_NAME}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    echo "Available configs:"
    ls configs/stage1/training/medical/*.yaml
    exit 1
fi

echo "=== Stage 1 Training: ${RUN_NAME} ==="
echo "Config: ${CONFIG}"
echo "Results: ${RESULTS_DIR}"

# Single GPU (use torchrun for multi-GPU)
python src/train_stage1.py \
    --config "$CONFIG" \
    --results-dir "$RESULTS_DIR" \
    --precision bf16

# For multi-GPU replace the above with:
# torchrun --standalone --nproc_per_node=8 \
#     src/train_stage1.py \
#     --config "$CONFIG" \
#     --results-dir "$RESULTS_DIR" \
#     --precision bf16 \
#     --wandb
