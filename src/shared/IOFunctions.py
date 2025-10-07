import os
def is_dm_format(file_path):
        _, file_extension = os.path.splitext(file_path)
        return file_extension in [".dm3", ".dm4"]

def is_tiff_format(file_path):
        _, file_extension = os.path.splitext(file_path)
        return file_extension in [".tif", ".tiff"]


def validate_file_extension(file_path, allowed_file_extensions):
        _, file_extension = os.path.splitext(file_path)
        return file_extension in allowed_file_extensions

