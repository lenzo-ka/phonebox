#!/usr/bin/env bash
#
# Train a PocketSphinx-compatible G2P model from CMUdict
#
# Quick method (recommended):
#   phonebox recipe cmudict pocketsphinx -o g2p.py
#
# This script shows the step-by-step process for learning/debugging:
#   1. Fetch CMUdict
#   2. Align letters to phonemes (EM algorithm)
#   3. Vectorize alignments (context windows)
#   4. Train decision tree
# Requires installed phonebox and network access for an initial CMUdict fetch.
# Usage: bash examples/train_pocketsphinx_g2p.sh [DATA_DIR] [MODELS_DIR]
# Retain DATA_DIR/cmudict/LICENSE; a full CMUdict run can take substantial time.
#

set -euo pipefail

# Configuration
DATA_DIR="${1:-data}"
MODELS_DIR="${2:-models}"
LOCALE="en_US"
WIDTH=7

# Output files
DICT_FILE="${DATA_DIR}/cmudict/cmudict.dict"
ALIGNMENTS_FILE="${DATA_DIR}/cmudict/alignments.txt"
VECTORS_FILE="${DATA_DIR}/cmudict/vectors.txt"
MODEL_FILE="${MODELS_DIR}/en_US_pocketsphinx.g2p.gz"

echo "=== PocketSphinx G2P Training Pipeline ==="
echo ""
echo "TIP: For production, use the one-liner instead:"
echo "  phonebox recipe cmudict pocketsphinx -o g2p.py"
echo ""

# Create output directories
mkdir -p "${DATA_DIR}" "${MODELS_DIR}"

# Step 1: Fetch CMUdict
if [[ -f "${DICT_FILE}" ]]; then
    echo "Step 1: CMUdict already exists at ${DICT_FILE}"
else
    echo "Step 1: Fetching CMUdict..."
    phonebox dict fetch cmudict --data-dir "${DATA_DIR}"
fi
echo ""

# Step 2: Align letters to phonemes
echo "Step 2: Aligning letters to phonemes..."
phonebox align "${DICT_FILE}" \
    --locale "${LOCALE}" \
    --phoneset cmu --width "${WIDTH}" \
    --remove-stress \
    -o "${ALIGNMENTS_FILE}"
echo ""

# Step 3: Vectorize alignments
echo "Step 3: Vectorizing alignments..."
phonebox vectorize "${ALIGNMENTS_FILE}" \
    --locale "${LOCALE}" \
    --phoneset cmu --width "${WIDTH}" \
    --remove-stress \
    -o "${VECTORS_FILE}"
echo ""

# Step 4: Train from vectors
echo "Step 4: Training decision tree..."
phonebox model train "${LOCALE}" \
    --vectors "${VECTORS_FILE}" \
    --phoneset cmu --width "${WIDTH}" \
    --remove-stress \
    --trainer native \
    -o "${MODEL_FILE}"
echo ""

# Show results
echo "=== Training Complete ==="
echo ""
echo "Files created:"
ls -lh "${MODEL_FILE}"
echo ""

# Inspect output; these examples are not an accuracy or correctness evaluation.
echo "Testing model..."
phonebox pronounce hello world phonebox -m "${MODEL_FILE}"
