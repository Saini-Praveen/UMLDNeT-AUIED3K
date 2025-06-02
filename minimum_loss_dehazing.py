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
def minimum_loss_dehazing(image, t_min=0.1, win_size=15):
    if image is None or image.size == 0:
        raise ValueError("Empty image data")
    
    # Convert image to float32 for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Split channels
    b, g, r = cv2.split(image_float)
    channels = [r, g, b]
    
    # Create output image container
    dehazed_channels = []
    
    # Process each channel separately
    for c in range(3):
        # Get current channel
        I_c = channels[c]
        
        # Calculate atmospheric light B^c (eq. 9)
        dark_channel = np.min(image_float, axis=2)
        flat_dark = dark_channel.flatten()
        top_indices = np.argsort(flat_dark)[-int(len(flat_dark) * 0.001):]  # Top 0.1% brightest
        B_c = np.mean(channels[c].flatten()[top_indices])
        
        # Create masks for x1 and x2 regions
        x1_mask = (I_c <= B_c)
        x2_mask = ~x1_mask
        
        # Calculate transmission map t^c(x) (eq. 8) - Vectorized
        t_c = np.zeros_like(I_c)
        
        # For x1 regions: t_c = 1 - min(min(I^c(y)))/B^c
        min_vals = cv2.erode(I_c, np.ones((win_size, win_size), np.uint8))
        t_c_x1 = 1.0 - min_vals / np.maximum(B_c, 0.01)
        
        # For x2 regions: t_c = (max(max(I^c(y)))-B^c)/(1-B^c)
        max_vals = cv2.dilate(I_c, np.ones((win_size, win_size), np.uint8))
        t_c_x2 = (max_vals - B_c) / np.maximum(1.0 - B_c, 0.01)
        
        # Combine transmission maps based on regions
        t_c[x1_mask] = t_c_x1[x1_mask]
        t_c[x2_mask] = t_c_x2[x2_mask]
        
        # Refine transmission map using guided filter
        t_c = cv2.ximgproc.guidedFilter(
            guide=image_float[:, :, c].astype(np.float32), 
            src=t_c.astype(np.float32), 
            radius=win_size, 
            eps=1e-3
        )
        
        # Ensure transmission map is within bounds
        t_c = np.clip(t_c, t_min, 1.0)
        
        # Apply initial image recovery (eq. 10)
        J_c = (I_c - B_c) / np.maximum(t_c, t_min) + B_c
        
        # AFDRM IMPLEMENTATION (eq. 6)
        J_c_mapped = np.copy(J_c)
        
        # Dark regions (x1): Map to [0, B^c]
        if np.any(x1_mask):
            J_min_x1 = np.min(J_c[x1_mask]) if np.any(x1_mask) else 0
            denom_x1 = B_c - J_min_x1
            if denom_x1 > 1e-6:  # Avoid division by zero or negative
                J_c_mapped[x1_mask] = B_c * (J_c[x1_mask] - J_min_x1) / denom_x1
            else:
                J_c_mapped[x1_mask] = J_c[x1_mask]  # Fallback to original values
        
        # Bright regions (x2): Map to [B^c, 1]
        if np.any(x2_mask):
            J_max_x2 = np.max(J_c[x2_mask]) if np.any(x2_mask) else 1
            denom_x2 = J_max_x2 - B_c
            if denom_x2 > 1e-6:  # Avoid division by zero
                J_c_mapped[x2_mask] = B_c + (1 - B_c) * (J_c[x2_mask] - B_c) / denom_x2
            else:
                J_c_mapped[x2_mask] = J_c[x2_mask]  # Fallback to original values
        
        # Ensure output is within valid range [0, 1]
        J_c_final = np.clip(J_c_mapped, 0.0, 1.0)
        dehazed_channels.append(J_c_final)
    
    # Combine channels
    dehazed_image = cv2.merge([dehazed_channels[2], dehazed_channels[1], dehazed_channels[0]])
    
    # Apply optimization to ensure constraints (eq. 4)
    dehazed_image = np.clip(dehazed_image, 0.0, 1.0)
    
    # Convert back to 8-bit
    dehazed_image = (dehazed_image * 255).astype(np.uint8)
    return dehazed_image

'''

