from PIL import Image
class dmFileReader():
    def __init__(self):
        return
    
    def get_image_from_dm_file(self, file_path):
        from ncempy.io import dm

        with dm.fileDM(file_path) as dmFile1: 
            tags = dmFile1.allTags
            high_limit = float(tags['.DocumentObjectList.1.ImageDisplayInfo.HighLimit'])
            low_limit = float(tags['.DocumentObjectList.1.ImageDisplayInfo.LowLimit'])
            file = dmFile1.getDataset(0)
            image_data = file['data']
            height, width = image_data.shape

            normalized_image = self.set_min_and_max(image_data, low_limit, high_limit)
            pil_image = Image.fromarray(normalized_image).convert('L')

            
            
            return pil_image
    
    def get_tensor_from_dm_file(self, file_path):
        from ncempy.io import dm

        import torchvision.transforms.functional as TF
        with dm.fileDM(file_path) as dmFile1: 
            tags = dmFile1.allTags
            high_limit = float(tags['.DocumentObjectList.1.ImageDisplayInfo.HighLimit'])
            low_limit = float(tags['.DocumentObjectList.1.ImageDisplayInfo.LowLimit'])

            file = dmFile1.getDataset(0)
            image_data = file['data']

            normalized_array = self.set_min_and_max(image_data, low_limit, high_limit)
            pil_image = Image.fromarray(normalized_array).convert('L')
            return TF.to_tensor(pil_image).unsqueeze(0)
        

    def normalize_image_data(self, image_data):
        image_data = image_data - image_data.min()
        image_data = image_data / image_data.max()
        image_data = image_data * 255
        return image_data
    
    def set_min_and_max(self, image_data, min_value, max_value):
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        import numpy as np

        clipped_image = np.clip(image_data, min_value, max_value)
        scaled_image = (clipped_image - min_value) / (max_value - min_value) * 255

        return scaled_image.astype("uint8")
    
    def get_pixel_size(self, file_path):
        from ncempy.io import dm

        with dm.fileDM(file_path) as dmFile1: 
            file = dmFile1.getDataset(0)
            pixel_unit = file['pixelUnit']
            pixel_size = file['pixelSize']
            return pixel_size, pixel_unit[0]
        