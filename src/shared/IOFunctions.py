import os
def is_dm_format(file_path):
        _, file_extension = os.path.splitext(file_path)
        return file_extension in [".dm3", ".dm4"]

def is_tiff_format(file_path):
        _, file_extension = os.path.splitext(file_path)
        return file_extension in [".tif", ".tiff"]