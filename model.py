import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEnhancementNet(nn.Module):
    def __init__(self):
        super(ImageEnhancementNet, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Additional convolution to process the second input (dehazed image)
        self.conv_dh = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn_dh = nn.BatchNorm2d(64)

        # Combining the first and dehazing conv outputs
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Rest of the network
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, dehazed_input):
        # Apply first convolution layer
        x1 = F.relu(self.bn1(self.conv1(x)))

        # Apply convolution on the dehazed input
        x2 = F.relu(self.bn_dh(self.conv_dh(dehazed_input)))

        # Concatenate the original and dehazed paths
        combined = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension

        # Apply the remaining layers
        x = F.relu(self.bn2(self.conv2(combined)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = self.conv4(x)
        x= self.sigmoid(self.conv4(x))# Output image

        return x
