import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import ImageEnhancementNet
from data import UnderwaterDataset
import torchvision.transforms as transforms
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']="1"

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return ssim(img1, img2, multichannel=True)

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
num_epochs = 100
input_dir = 'Dataset/Raw'  # Path to images
target_dir = 'Dataset/Reference'  # Path to ground truth images

# Data transforms (resize to 256x256 and normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the full dataset
full_dataset = UnderwaterDataset(input_dir=input_dir, target_dir=target_dir, transform=transform)

# Split the dataset into training and validation sets (800 for train, 90 for validation)
train_size = 2700
val_size = 300
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = ImageEnhancementNet()
criterion = nn.MSELoss()  # Mean Squared Error for pixel-wise difference
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Function to evaluate the model (for validation)
def evaluate(loader, model):
    model.eval()  # Set model to evaluation mode
    mse_total, ssim_total, psnr_total = 0.0, 0.0, 0.0

    with torch.no_grad():  # Disable gradient computation during validation
        for i, data in tqdm(enumerate(loader), total=len(loader), desc="Validating", leave=False):
            inputs, dehazed_inputs, targets = data
            inputs, dehazed_inputs, targets = inputs.to(device), dehazed_inputs.to(device), targets.to(device)

            # Forward pass with two inputs (original and dehazed)
            outputs = model(inputs, dehazed_inputs)
            # Convert outputs and targets to NumPy arrays for metric calculations
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for b in range(outputs_np.shape[0]):  # Loop over each image in the batch
                output_img = (outputs_np[b].transpose(1, 2, 0) * 255).astype(np.uint8)
                target_img = (targets_np[b].transpose(1, 2, 0) * 255).astype(np.uint8)

                # Calculate PSNR and SSIM
                psnr_value = calculate_psnr(output_img, target_img)
                ssim_value = calculate_ssim(output_img, target_img)

                # Accumulate values
                mse_total += np.mean((output_img - target_img) ** 2)
                psnr_total += psnr_value
                ssim_total += ssim_value

    # Average metrics
    num_samples = len(loader.dataset)
    mse_avg = mse_total / num_samples
    psnr_avg = psnr_total / num_samples
    ssim_avg = ssim_total / num_samples

    return mse_avg, psnr_avg, ssim_avg

# Create a models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Prepare CSV files and write headers 
np.savetxt('models/trainmetrics.csv', np.array([]), header='Epoch,Training_Loss,Training_MSE,Training_PSNR,Training_SSIM', delimiter=',', comments='')
np.savetxt('models/valmetrics.csv', np.array([]), header='Epoch,Validation_MSE,Validation_PSNR,Validation_SSIM', delimiter=',', comments='')

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}")

    # Train the model for one epoch
    for i, data in train_progress_bar:
        inputs, dehazed_inputs, targets = data
        inputs, dehazed_inputs, targets = inputs.to(device), dehazed_inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass with two inputs
        outputs = model(inputs, dehazed_inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Training metrics (calculated after each epoch)
    #train_mse, train_psnr, train_ssim = evaluate(train_loader, model)

    # Validation metrics
    val_mse, val_psnr, val_ssim = evaluate(val_loader, model)

    # Print results after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Training Loss: {running_loss / len(train_loader):.4f}')
    #print(f'Training MSE: {train_mse:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}')
    print(f'Validation MSE: {val_mse:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}')

    # Save training metrics to CSV after each epoch
    #train_metrics = np.array([[epoch+1, running_loss / len(train_loader), train_mse, train_psnr, train_ssim]])
    val_metrics = np.array([[epoch+1, val_mse, val_psnr, val_ssim]])
    '''
    with open('models/trainmetrics.csv', 'ab') as f:
        np.savetxt(f, train_metrics, delimiter=',', fmt='%.4f')
    '''
    with open('models/valmetrics.csv', 'ab') as f:
        np.savetxt(f, val_metrics, delimiter=',', fmt='%.4f')

# Save the trained model
torch.save(model.state_dict(), 'models/image_enhancement_model.pth')
