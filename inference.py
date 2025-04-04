import os
from PIL import Image
import numpy as np
import torch
from model import ImageEnhancementNet
from minimum_loss_dehazing import minimum_loss_dehazing  # Assuming this is implemented
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES']="1"

# Define source and output directories
source_dir = 'Dataset/Raw'  # Folder containing the input images
output_dir = 'outputs'  # Folder where output images will be saved

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define image transformation (same as used in training)
transform = transforms.Compose([
    #transforms.Resize((256, 256)),  # Ensure the images are resized as needed
    transforms.ToTensor(),  # Convert to tensor
])

# Load the model (ensure the model architecture is the same as in training)
model = ImageEnhancementNet()
model.load_state_dict(torch.load('models/image_enhancement_model.pth'))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Process each image in the source directory
for image_name in os.listdir(source_dir):
    # Load input image from the source directory
    input_image_path = os.path.join(source_dir, image_name)
    input_image = Image.open(input_image_path)
    input_image = np.array(input_image)

    # Apply dehazing using your minimum loss dehazing function
    #dehazed_image = minimum_loss_dehazing(input_image)
    dehazed_image = minimum_loss_dehazing(image_name)

    # Convert the original input and dehazed image to tensors
    input_tensor = transform(Image.fromarray(input_image)).unsqueeze(0).to(device)
    dehazed_tensor = transform(Image.fromarray(dehazed_image)).unsqueeze(0).to(device)

    # Inference using the model
    with torch.no_grad():
        output_tensor = model(input_tensor, dehazed_tensor)

    # Convert the output tensor back to an image
    output_image = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_image = (output_image * 255).astype(np.uint8)  # Scale back to [0, 255]

    # Save the output image to the output directory with the same filename
    output_image_path = os.path.join(output_dir, image_name)
    output_img = Image.fromarray(output_image)
    output_img.save(output_image_path)

    print(f"Processed and saved: {output_image_path}")

print("Inference completed for all images.")