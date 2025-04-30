from src.shared.Commands import Command
from src.model.RequestHandler import request_handler
from src.model.UNet import UNet

class Controller():
    def __init__(self, pre_loaded_model_name=None):
        self.unet = UNet(f"src/data/model/{pre_loaded_model_name}")
        self.request_handler = request_handler(self.unet)
        
        self.commands = {
            Command.SEGMENT: self.request_handler.process_request_segment,
            Command.RETRAIN: self.request_handler.process_request_train,
            Command.LOAD_MODEL: self.request_handler.process_request_load_model,
            Command.TEST_MODEL: self.request_handler.process_request_test_model,
            Command.SEGMENT_FOLDER: self.request_handler.process_request_segment_folder,
            Command.LOAD_IMAGE: self.request_handler.process_request_load_image,
            # Command.ANALYZE: self.request_handler.analyze_segmentation,
            # Command.EXPORT: self.request_handler.export_results
        }
    
    def process_command(self, command, *args, **kwargs):
        """Routes the command to the appropriate backend method."""
        if command not in self.commands:
            print(f"Unknown command: {command}")
            return
        
        # try:
        data = self.commands[command](*args, **kwargs)
        return data
        # except Exception as e:
        #     print(e)
        #     return None

            
            