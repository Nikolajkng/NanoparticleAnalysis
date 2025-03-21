class ScaleInfo():
    def __init__(self, start_x, end_x, real_scale_length, image_width):
        self.start_x: int = start_x
        self.end_x: int = end_x
        self.real_scale_length: float = real_scale_length
        self.image_width: int = image_width