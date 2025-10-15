from PIL import Image
import threading
from src.model.PlottingTools import *
class RequestHandler:
    def __init__(self, pre_loaded_model_name=None):
        self.unet = None
        self.model_ready_event = threading.Event()
        self.load_model_async(pre_loaded_model_name)

    def load_model_async(self, model_name):
        def load():
            from src.model.UNet import UNet
            self.unet = UNet(pre_loaded_model_path=f"src/data/model/{model_name}")  # or UNet(pre_loaded_model_path=...)
            self.model_ready_event.set()
            print("Model ready")

        threading.Thread(target=load, daemon=True).start()
        
    def process_request_train(self, model_config, stop_training_event = None, loss_callback = None, test_callback = None):  
        from src.model.CrossValidation import cv_holdout
        from src.model.UNet import UNet
        self.model_ready_event.wait()
        self.unet = UNet()
        iou, dice_score = cv_holdout(self.unet, model_config, self.unet.preferred_input_size, stop_training_event, loss_callback, test_callback)
        print(f"Model IOU: {iou}\nModel Dice Score: {dice_score}")
        return iou, dice_score

    def process_request_segment(self, image, output_folder, return_stats=False):
        """
        Process an image through the segmentation pipeline.
        
        Args:
            image: The input image to segment
            output_folder: Folder to save the statistics
            return_stats: Whether to include raw statistics in the return value
            
        Returns:
            Tuple containing:
            - Segmented image (PIL Image)
            - Annotated image (PIL Image) 
            - Table data
            - Histogram figure
            - Stats (optional, only if return_stats is True)
        """
        from src.model.DataTools import ImagePreprocessor
        from src.model.SegmentationAnalyzer import SegmentationAnalyzer

        # Wait for model to be ready
        self.model_ready_event.wait()
                
        # 1. Initialize preprocessor
        preprocessor = ImagePreprocessor(self.unet.preferred_input_size)
        
        # 2. Prepare image patches
        tensor, tensor_mirror_filled, patches, stride_length = preprocessor.prepare_image_patches(
            image.pil_image, 
            self.unet.device
        )
        
        # 3. Run model inference
        segmentations = self._run_model_inference(patches, tensor)
        
        # 4. Post-process segmentation output
        segmented_image_2d = preprocessor.post_process_segmentation(
            segmentations,
            tensor_mirror_filled,
            tensor,
            stride_length
        )
        
        # 5. Analyze results and generate statistics
        analyzer = SegmentationAnalyzer()
        results = analyzer.analyze_segmentation(segmented_image_2d, image.file_info, output_folder)
        
        if return_stats:
            return results
        else:
            return results[:-1]  # Return without stats
    
    def _run_model_inference(self, patches, tensor):
        """
        Run model inference on image patches.
        
        Args:
            patches: Array of image patches
            tensor: Original tensor for dtype/device info
            
        Returns:
            numpy array: Model predictions
        """
        import torch
        
        # Convert patches to tensor
        patches_tensor = torch.tensor(patches, dtype=tensor.dtype, device=tensor.device)
        
        # Optimize model for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet.to(device, memory_format=torch.channels_last).eval()
        
        # Create optimized model for inference
        scripted_unet = torch.jit.script(self.unet)
        scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
        
        # Prepare input tensor
        patches_tensor = patches_tensor.to(device, memory_format=torch.channels_last)
        
        # Run inference with appropriate precision
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.cuda.amp.autocast("cuda"):
                    output = scripted_unet(patches_tensor)
            else:
                output = scripted_unet(patches_tensor)
        
        # Convert to numpy for post-processing
        return output.cpu().detach().numpy()   

    def process_request_load_model(self, model_path):
        self.model_ready_event.wait()
        self.unet.load_model(model_path)
        return None
    
    def process_request_test_model(self, test_data_image_dir, test_data_mask_dir, testing_callback = None):
        from src.model.SegmentationDataset import SegmentationDataset
        from torch.utils.data import DataLoader


        dataset = SegmentationDataset(test_data_image_dir, test_data_mask_dir)
        test_dataloader = DataLoader(dataset, batch_size=1)
        from src.model.ModelEvaluator import ModelEvaluator

        self.model_ready_event.wait()
        iou, dice_score = ModelEvaluator.evaluate_model(self.unet, test_dataloader, testing_callback)
        print(iou)
        print(dice_score)
        return iou, dice_score
        
    def process_request_segment_folder(self, input_folder, output_parent_folder):
        """
        Process all images in a folder through the segmentation pipeline.
        
        Args:
            input_folder: Path to folder containing images to process
            output_parent_folder: Path to folder where results will be saved
        """
        from src.model.BatchProcessor import BatchProcessor
        
        self.model_ready_event.wait()
        
        batch_processor = BatchProcessor()
        batch_processor.process_folder(
            input_folder,
            output_parent_folder,
            self.process_request_segment
        )
        
    def process_request_load_image(self, image_path):
        """
        Load and preprocess an image for segmentation.
        
        Args:
            image_path: Path to the image file to load
            
        Returns:
            ParticleImage: The loaded and preprocessed image
        """
        from src.shared.ParticleImage import ParticleImage
        return ParticleImage.load_and_preprocess(image_path)