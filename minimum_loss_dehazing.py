import cv2
import numpy as np
import os
from PIL import Image

def minimum_loss_dehazing(image):
   dehazed_dir ='MLD/'
   dehazed_image = Image.open(os.path.join(dehazed_dir, image))
   dehazed_image = np.array(dehazed_image)
    #input_image = cv2.imread(image)
    
   return dehazed_image

'''
def minimum_loss_dehazing(image, t_min=0.1, omega=0.95, win_size=15):
    if image is None or image.size == 0:
        raise ValueError("Empty image data")
    
    # Convert image to float32 for processing
    image_float = image.astype(np.float32) / 255
    
    # Estimate atmospheric light using the dark channel prior
    dark_channel = np.min(image_float, axis=2)
    bright_pixel = np.unravel_index(np.argmax(dark_channel), dark_channel.shape)
    
    # Ensure window for atmospheric light calculation does not exceed image bounds
    h, w, _ = image.shape
    win_start_y = max(bright_pixel[0] - win_size, 0)
    win_end_y = min(bright_pixel[0] + win_size, h)
    win_start_x = max(bright_pixel[1] - win_size, 0)
    win_end_x = min(bright_pixel[1] + win_size, w)
    
    atmospheric_light = np.mean(image_float[win_start_y:win_end_y, win_start_x:win_end_x], axis=(0, 1))
    
    # Estimate the transmission map
    transmission_map = 1 - omega * dark_channel
    
    # Refine transmission map using guided filter
    transmission_map = cv2.ximgproc.guidedFilter(
        guide=image, src=transmission_map.astype(np.float32), radius=win_size, eps=1e-3
    )
    
    # Apply Minimum Loss Dehazing
    transmission_map = np.clip(transmission_map, t_min, 1)
    
    # Avoid division by zero in transmission map
    transmission_map[transmission_map == 0] = t_min
    
    dehazed_image = (image_float - atmospheric_light) / transmission_map[:, :, np.newaxis] + atmospheric_light
    
    # Convert back to 8-bit and return the result
    dehazed_image = np.clip(dehazed_image * 255, 0, 255).astype(np.uint8)
    
    return dehazed_image
'''

