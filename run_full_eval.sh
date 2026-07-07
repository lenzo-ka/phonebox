#!/usr/bin/env bash
# Sequential batch: regenerate G2P_COMPARE.md, sweep span x lm-order, dump units.
set -euo pipefail

cd "$(dirname "${0}")"

: "${PHONEDECODING_LEXICON_DIR:?set PHONEDECODING_LEXICON_DIR}"
: "${PHONEDECODING_G2P_DIR:?set PHONEDECODING_G2P_DIR}"

echo "=== [1/3] Re-run compare_g2p_all (new joins, add-k LM) ==="
python -u compare_g2p_all.py --parallel-align --output docs/G2P_COMPARE.md

echo
echo "=== [2/3] Sweep span x lm-order on fr/de/pt/en ==="
python -u compare_g2p_sweep.py \
    --locales fr_FR de_DE pt_BR en_US \
    --letter-spans 2 3 \
    --lm-orders 2 3 \
    --parallel-align \
    --output docs/G2P_SWEEP.md

echo
echo "=== [3/3] Dump top multigram units for all locales ==="
python -u dump_units.py --top 30 --output docs/G2P_UNITS.md

echo
echo "=== Optional: join-off fair compare (slow; trains 1:1 per locale) ==="
echo "  python -u compare_g2p_all.py --no-config-joins --parallel-align --output docs/G2P_COMPARE_NO_JOINS.md"

echo
echo "All done."
