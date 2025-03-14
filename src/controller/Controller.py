from shared.Commands import Command
from model.RequestHandler import request_handler
from model.UNet import UNet

class Controller():
    def __init__(self):
        self.unet = UNet()
        self.request_handler = request_handler(self.unet)
        
        self.commands = {
            Command.SEGMENT: self.request_handler.process_request_segment,
            Command.RETRAIN: self.request_handler.process_request_train,
            Command.LOAD_MODEL: self.request_handler.process_request_load_model,
            Command.CALCULATE_REAL_IMAGE_WIDTH: self.request_handler.process_request_calculate_image_width,
            # Command.ANALYZE: self.request_handler.analyze_segmentation,
            # Command.EXPORT: self.request_handler.export_results
        }
    
    def process_command(self, command, *args, **kwargs):
        """Routes the command to the appropriate backend method."""
        if command in self.commands:
            data, return_code = self.commands[command](*args, **kwargs)
            if return_code == 0:
                return data
            else:
                print(data)
                return None
        else:
            print(f"Unknown command: {command}")
            
            