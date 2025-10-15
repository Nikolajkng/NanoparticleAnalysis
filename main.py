import sys
import time
from PyQt5.QtWidgets import QApplication

import threading

def preload_torch():
    from ncempy.io import dm
    import numpy as np
    import torch
    import torchvision
    _ = torch.Tensor([0])  # Force lazy CUDA init

def main():
    threading.Thread(target=preload_torch, daemon=True).start()
    from src.gui.windows.MainWindow import MainWindow
    
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.MainWindow.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    #from src.model.CrossValidation import cv_kfold
    #cv_kfold("data/medres_images", "data/medres_masks")