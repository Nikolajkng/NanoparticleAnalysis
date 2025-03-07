from model.UNet import UNet
from shared.Commands import Command
class Controller():
    def __init__(self):
        self.unet = UNet()

        self.commands = {
            Command.SEGMENT: self.unet.process_request_segment,
            Command.RETRAIN: self.unet.process_request_train,
            # Command.ANALYZE: self.backend.analyze_segmentation,
            # Command.EXPORT: self.backend.export_results
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