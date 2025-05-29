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

    # Generate a random input tensor (1 sample, 1 channel, 256x256 image)
    input_tensor = load_image_as_tensor("data/UNET_visualizations/input.PNG")
    input_tensor = input_tensor.squeeze(0) 
    
    image_width, image_height = input_tensor.shape[-2:]
    #start_height = np.random.randint(0, image_height - 256)
    #start_width = np.random.randint(0, image_width - 256)
    #cropped_image = input_tensor[:, start_width:start_width + 256, start_height:start_height + 256]
    cropped_image = input_tensor.unsqueeze(0)  # Add batch dimension
    # Pass the input tensor through the model
    output = model(cropped_image)

    # Visualize the final output
    #model._visualize_feature_map(output.argmax(dim=1).unsqueeze(0), "Final Output", is_output=True)
    showTensor(output.argmax(dim=1))
    #showTensor(cropped_image)
if __name__ == "__main__":
    main()
    #cv_kfold("data/medres_images", "data/medres_masks")