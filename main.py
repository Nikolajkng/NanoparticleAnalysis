import sys
import time
from PyQt5.QtWidgets import QApplication

import threading

def preload_torch():
    import numpy as np
    import torch
    import torchvision
    _ = torch.Tensor([0])  # Force lazy CUDA init
    import ncempy
    import sklearn

def main():
    start_time = time.perf_counter()
    threading.Thread(target=preload_torch, daemon=True).start()
    from src.gui.windows.MainWindow import MainWindow
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.MainWindow.show()

    end_time = time.perf_counter()
    print(f"Startup time: {end_time - start_time:.4f} seconds")

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    #cv_kfold("data/medres_images", "data/medres_masks")