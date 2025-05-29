import torch
from src.model.UNet import UNet  # Directly import the UNet class
from src.model.DataTools import load_image_as_tensor
from src.model.DataTools import showTensor
import numpy as np
def main():
    # Create an instance of the UNet model
    model = UNet("src/data/model/UNet_best_09-05.pt")

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