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


#from src.model.CrossValidation import cv_kfold
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
    # import subprocess

    # threshold_us = 1_000_00  # 1000 ms

    # result = subprocess.run(
    #     [sys.executable, "-X", "importtime", "-c", "from src.gui.windows.MainWindow import MainWindow"],
    #     capture_output=True, text=True
    # )

    # for line in result.stderr.splitlines():
    #     parts = line.strip().split('|')
    #     if len(parts) == 3:
    #         try:
    #             cumulative_us = int(parts[1].strip())
    #             if cumulative_us > threshold_us:
    #                 print(line)
    #         except ValueError:
    #             pass