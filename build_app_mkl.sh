#!/bin/bash

# === CONFIG ===
PY_SCRIPT="main.py"               
OUTPUT_NAME="my_app"                 

# === DETECT ACTIVE CONDA ENV LIB FOLDER ===
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "[!] No active Conda environment detected."
    echo "    Activate a Conda environment first, then rerun this script."
    exit 1
fi

ENV_LIB_DIR="$CONDA_PREFIX/lib"
echo "[+] Using Conda environment at: $CONDA_PREFIX"
echo "[+] Searching for MKL libraries in: $ENV_LIB_DIR"

# === FIND MKL LIBRARIES ===
MKL_LIBS=$(find "$ENV_LIB_DIR" \( -name "libmkl*.so*" -o -name "libcblas.so*" \) 2>/dev/null)

if [[ -z "$MKL_LIBS" ]]; then
    echo "[!] No MKL libraries found. Aborting."
    exit 1
fi

# === BUILD --add-binary FLAGS ===
ADD_BINARIES=""
for lib in $MKL_LIBS; do
    ADD_BINARIES+=" --add-binary \"$lib:.\""
done

# === RUN PYINSTALLER ===
echo "[+] Building with PyInstaller..."
MODEL_PATH="src/data/model/UNet_256_downsized.pt"
ADD_DATA="--add-data \"src/data/model:src/data/model\""

eval pyinstaller "$PY_SCRIPT" \
    --onefile \
    --name "$OUTPUT_NAME" \
    $ADD_DATA \
    $ADD_BINARIES

echo "[✓] Build complete. Executable is in dist/$OUTPUT_NAME"
