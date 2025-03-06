from model.UNet import UNet
from shared.Commands import Command
class Controller():
    def __init__(self):
        self.unet = UNet()

        self.commands = {
            Command.SEGMENT: self.unet.forward,
            Command.RETRAIN: self.unet.process_request_train,
            # Command.ANALYZE: self.backend.analyze_segmentation,
            # Command.EXPORT: self.backend.export_results
        }
    
    def process_command(self, command, *args, **kwargs):
        """Routes the command to the appropriate backend method."""
        if command in self.commands:
            return_code = self.commands[command](*args, **kwargs)
            if return_code == 0:
                return "We good"
            else:
                return "Something went wrong"
        else:
            print(f"Unknown command: {command}")