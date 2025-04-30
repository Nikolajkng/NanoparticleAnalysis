import os
from PIL import Image
import tifffile
from shared.IOFunctions import is_dm_format, is_tiff_format
from shared.FileInfo import FileInfo
from model.dmFileReader import dmFileReader
class ParticleImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.pil_image = self.load_image(image_path)
        self.file_info = None
        self.convert_from_cm = True
        

    def load_image(self, image_path):
        if is_dm_format(image_path):
            reader = dmFileReader()
            return  reader.get_image_from_dm_file(image_path)
        else:
            return Image.open(image_path).convert("L")
    
    def get_file_info(self, file_path):
        file_info = FileInfo()

        if is_dm_format(file_path):
            reader = dmFileReader()
            pixel_size, pixel_unit = reader.get_pixel_size(file_path)
        elif is_tiff_format(file_path):
            pixel_size, pixel_unit = self.extract_pixel_size_from_tiff_file(file_path)

        file_info.pixel_height = pixel_size[0]
        file_info.pixel_width = pixel_size[1]
        file_info.unit = pixel_unit
        file_info.real_height = float(file_info.pixel_height*self.pil_image.height)
        file_info.real_width = float(file_info.pixel_width*self.pil_image.width)
        file_info.width = self.pil_image.width 
        file_info.height = self.pil_image.height
        split_name = os.path.splitext(os.path.basename(file_path))
        file_info.file_name = split_name[0]
        file_info.file_type = split_name[1]

        return file_info
    
    def extract_pixel_size_from_tiff_file(self, file_path):
        try:
            with tifffile.TiffFile(file_path) as tif:
                    tags = tif.pages[0].tags
                    #tvips = tags.get('TVIPS') # Can only extract pixel size if file is from TVIPS
                    
                    X_Resolution_fraction = tags.get('XResolution')
                    Y_Resolution_fraction = tags.get('YResolution')
                    if not X_Resolution_fraction or not Y_Resolution_fraction:
                        pixel_width = 1.0
                        pixel_height = 1.0
                        unit_string = "pixel"
                        return (pixel_width, pixel_height), unit_string
                    
                    X_Resolution_float = self.get_float_from_fraction(X_Resolution_fraction.value)
                    Y_Resolution_float = self.get_float_from_fraction(Y_Resolution_fraction.value)
                    pixel_width = None
                    pixel_height = None
                    if X_Resolution_fraction != 0: 
                        pixel_width = 1/X_Resolution_float
                        pixel_height = 1/Y_Resolution_float

                    unit = tags.get('ResolutionUnit').value
                    unit_string = None
                    if unit == 1:
                        description = tags.get('ImageDescription').value
                        if description:
                            if "nm" in description:
                                unit_string = "nm"
                            elif "\\u00B5m" in description: #micrometer
                                unit_string = "\u00B5m"
                            elif "mm" in description:
                                unit_string = "mm"
                            elif "cm" in description:
                                unit_string = "cm"
                            elif "inch" in description:
                                unit_string = "inch"
                            else:
                                unit_string = " "
                    elif unit == 2:
                        if X_Resolution_float == 72:
                            pixel_width = 1.0
                            pixel_height = 1.0
                            unit_string = "pixel"
                        else:
                            unit_string = "inch"
                    elif unit == 3:
                        unit_string = "cm"

                    if self.convert_from_cm and unit_string == "cm":
                        pixel_width = pixel_width * 10000
                        pixel_height = pixel_height * 10000
                        unit_string = "\u00B5m"

                    

                    # if tvips:
                    #     pixel_width = tvips.value['PixelSizeX']
                    #     pixel_height = tvips.value['PixelSizeY']
                    
                    return (pixel_width, pixel_height), unit_string
        except (tifffile.TiffFileError, TypeError):
            return None
    
    def get_float_from_fraction(self, fraction):
        if fraction[1] == 0:
            return 0.0
        return float(fraction[0]) / float(fraction[1])
    
    def resize(self, new_size: tuple[int, int]):
        self.pil_image.thumbnail(new_size)

        self.file_info.pixel_width = self.file_info.pixel_width * self.file_info.width / self.pil_image.width
        self.file_info.pixel_height = self.file_info.pixel_height * self.file_info.height / self.pil_image.height
        self.file_info.width, self.file_info.height = self.pil_image.width, self.pil_image.height

