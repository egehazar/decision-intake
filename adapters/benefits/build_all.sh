#!/usr/bin/env bash
# Build the whole outcome table: NCHUNKS chunks, PAR processes at a time.
set -u
cd "$(dirname "$0")/../.."
# On Windows, PolicyEngine's longest file path exceeds MAX_PATH unless policyengine_us sits under a short
# prefix: copy the installed package to e.g. C:/pe and set PE_SHORT_PATH to it. Elsewhere leave it unset.
if [ -n "${PE_SHORT_PATH:-}" ]; then export PYTHONPATH="$PE_SHORT_PATH"; fi
N=${1:-30}; PAR=${2:-5}
mkdir -p data/cache/chunks
seq 0 $((N-1)) | xargs -P "$PAR" -I{} sh -c '[ -s data/cache/chunks/c{}.csv ] || .venv/Scripts/python.exe adapters/benefits/build_table.py {} '"$N"' data/cache/chunks/c{}.csv 2>&1 | grep -E "^chunk|Error|Traceback" '
echo "done: $(ls data/cache/chunks/*.csv | wc -l) of $N chunks"
