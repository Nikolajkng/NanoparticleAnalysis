#!/bin/bash

APP_NAME="NP"
ENTRY_SCRIPT="main.py"
MODEL_SOURCE="src/data/model/UNet_best_09-05.pt"
MODEL_DEST="src/data/model"

echo "🔧 Building macOS .app with PyInstaller..."

# Clean previous builds
rm -rf build dist "$APP_NAME.spec"

# Run PyInstaller with model included
pyinstaller --windowed --onefile \
  --name "$APP_NAME" \
  --add-data "$MODEL_SOURCE:$MODEL_DEST" \
  "$ENTRY_SCRIPT"

# Remove macOS quarantine flag so app can be opened
APP_BUNDLE="dist/$APP_NAME.app"
if [ -d "$APP_BUNDLE" ]; then
  echo "✅ App built successfully: $APP_BUNDLE"
  echo "🚫 Removing macOS quarantine attribute..."
  xattr -rd com.apple.quarantine "$APP_BUNDLE"
else
  echo "❌ Build failed. App not found."
  exit 1
fi

echo "🎉 Done! You can
