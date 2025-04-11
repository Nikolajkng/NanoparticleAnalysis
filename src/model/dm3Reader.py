from ncempy.io import dm

# Load the .dm3 file
filename = 'sample_0001.dm4'
reader = dm.dmReader(filename)
image_data = reader['data']

height, width = image_data.shape

# Metadata
pixel_unit = reader['pixelUnit']
pixel_size = reader['pixelSize']
print(f"{pixel_size[0]*height} {pixel_unit[0]}, {pixel_size[1]*width} {pixel_unit[1]}")


import tifffile
import struct

def extract_pixel_size(binary_data):
    # Define the key you're looking for
    key = "Pixel Size (um)"
    
    # Decode the binary string into a human-readable string
    decoded_data = binary_data.decode('utf-8', errors='ignore')
    
    # Find the position of the key "Pixel Size (um)"
    key_pos = decoded_data.find(key)
    
    if key_pos != -1:
        # The data after the key is the value you need (may need adjustment based on structure)
        start_pos = key_pos + len(key)+2+12
        
        # Extract a portion of the string right after the key (might need further adjustments)
        pixel_data_raw = binary_data[start_pos:start_pos+64]  # Take 4 bytes for float32
        
        # Check if it's a valid number in float format (try to unpack as float32)
        try:
            pixel_size_new = struct.unpack('f', pixel_data_raw)[0]  # Unpack as float32
            return pixel_size_new
        except struct.error:
            print(f"Error unpacking the byte data: {pixel_data_raw}")
            return None
    else:
        print(f"Key '{key}' not found in the binary data.")
        return None

filename = '../Data/A. sample_0001.tif'
with tifffile.TiffFile(filename) as tif:
    image = tif.pages[0].asarray()
    tags = tif.pages[0].tags
    height, width = image.shape
    print(f"Image dimensions: {width} x {height} pixels")
    extract_pixel_size(tags.get(65027).value)
