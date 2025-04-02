#!/bin/bash

# Navigate to src/gui/ui_designs
cd gui/ui_designs || { echo "ui_designs directory not found"; exit 1; }

# Execute the pyuic5 command
pyuic5 gui_prototype1.2_graphics_view.ui -o ../ui/MainUI.py

# Print success message
echo "UI file successfully converted to Python script."