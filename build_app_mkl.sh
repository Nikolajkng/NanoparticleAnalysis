#!/bin/bash

# === CONFIG ===
PY_SCRIPT="main.py"
OUTPUT_NAME="NanoAnalyzer"
MODEL_FILE="src/data/model/UNet_best_06-06.pt"
MODEL_DEST="src/data/model"

# === CHECK CONDA ENVIRONMENT ===
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "[!] No active Conda environment detected."
    echo "    Activate a Conda environment first, then rerun this script."
    exit 1
fi

echo "[+] Using Conda environment at: $CONDA_PREFIX"

# === BUILD PYINSTALLER COMMAND ===
echo "[+] Building with PyInstaller..."

pyinstaller --noconfirm --noconsole \
    --name "$OUTPUT_NAME" \
    --add-data "$MODEL_FILE:$MODEL_DEST" \
    "$PY_SCRIPT"

echo "[✓] Build complete. Executable is in dist/$OUTPUT_NAME"
