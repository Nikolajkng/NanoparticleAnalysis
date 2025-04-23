import os
class Segmenter:
    def __init__(self):
        """
        Initialize the Segmenter class.
        Add any necessary attributes here.
        """
        pass

    def segment_patch_batch(self, model_state_dict, device, patch_batch):
        print("Hello")
        import torch
        from model.UNet import UNet
        model = UNet()
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        segmented_outputs = []
        with torch.no_grad():
            for image in patch_batch:
                tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
                output = model(tensor)
                segmentation_numpy = output.detach().numpy()
                segmented_outputs.append(segmentation_numpy)
                print(f"Done {os.getpid()}")

        return segmented_outputs
    
    
    def segment(self, model_state_dict, images, result_array, start_index, dtype, device):
        import torch
        from model.UNet import UNet
        model = UNet()
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        image_idx = start_index
        for image in images:
            with torch.no_grad():
                segmentation = model(torch.tensor(image, dtype=dtype, device=device).unsqueeze(0))
            segmentation_numpy = segmentation.detach().numpy()
            result_array[image_idx] = segmentation_numpy
            print(f"Done {image_idx}")
            image_idx += 1
        