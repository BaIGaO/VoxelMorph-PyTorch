import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable # Variable is deprecated.
import numpy as np

# --- Modern Practice: Define device once ---
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(42)

#================================================================================
# 1. ATTENTION MODULES (Corrected)
#================================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid() # Use nn.Sigmoid for clarity

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention_map = self.sigmoid(avg_out + max_out)      
        return x * attention_map

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Ensure padding is calculated correctly
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(out))
        return x * attention_map

#================================================================================
# 2. BUILDING BLOCKS (Residual, ASPP are mostly fine)
#================================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        rates = [6, 12, 18]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
            
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.final(out)
        return out

#================================================================================
# 3. UNET ARCHITECTURE (Refactored)
#================================================================================

# --- New Block to correctly apply attention ---
class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = self.res_block(x)
        x = self.ca(x)
        x = self.sa(x)
        return x
        

class UNet(nn.Module):
    # --- Helper functions are fine, but let's make the main class cleaner ---
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = AttentionResidualBlock(in_channels=in_channel, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = AttentionResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = AttentionResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = ResidualBlock(256, 512) # No attention on the deepest layer is a common choice

        # Bottleneck
        self.bottleneck = ASPP(512, 512) # ASPP out_channels should match for concatenation

        # Decode
        # Note: This expansive block design is unconventional.
        # A more common approach is: Upsample -> Concat -> ConvBlock.
        # This implementation does: Concat -> ConvBlock -> Upsample.
        self.conv_decode4 = self._expansive_block(512 + 512, 512, 256)
        self.conv_decode3 = self._expansive_block(256 + 256, 256, 128)
        self.conv_decode2 = self._expansive_block(128 + 128, 128, 64)
        self.final_layer = self._final_block(64 + 64, 64, out_channel)

    def _expansive_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, mid_channels),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _final_block(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, mid_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False) # Use 1x1 conv for final mapping
        )

    def crop_and_concat(self, upsampled, bypass):
        # This function might not be necessary if input sizes are powers of 2
        # but it's good practice to keep it for robustness.
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        e1 = self.conv_encode1(x)
        p1 = self.pool1(e1)
        e2 = self.conv_encode2(p1)
        p2 = self.pool2(e2)
        e3 = self.conv_encode3(p2)
        p3 = self.pool3(e3)
        e4 = self.conv_encode4(p3)

        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decode
        d4 = self.conv_decode4(torch.cat((b, e4), dim=1))
        d3 = self.conv_decode3(torch.cat((d4, e3), dim=1))
        d2 = self.conv_decode2(torch.cat((d3, e2), dim=1))
        d1 = self.final_layer(torch.cat((d2, e1), dim=1))

        return d1

#================================================================================
# 4. SPATIAL TRANSFORMER (Refactored for Simplicity and Channels-First)
#================================================================================

class SpatialTransformation(nn.Module):
    def __init__(self):
        super(SpatialTransformation, self).__init__()
        # Smoothing layer for the deformation field
        self.smoothing = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, moving_image, deformation_field):
        # Input shapes are expected to be [B, C, H, W]
        # deformation_field should have 2 channels (dx, dy)
        
        # Smooth the deformation field
        deformation_field = self.smoothing(deformation_field)

        # Create a base grid
        b, _, h, w = deformation_field.shape
        # The new, efficient way to create a meshgrid
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 2).float() # Shape: [H, W, 2]
        grid = grid.unsqueeze(0).to(deformation_field.device) # Shape: [1, H, W, 2]
        
        # The deformation field is an offset to the base grid.
        # It needs to be permuted from [B, 2, H, W] to [B, H, W, 2] for addition
        new_grid = grid + deformation_field.permute(0, 2, 3, 1)

        # Normalize the grid to [-1, 1] for grid_sample
        new_grid[..., 0] = 2 * new_grid[..., 0] / (w - 1) - 1
        new_grid[..., 1] = 2 * new_grid[..., 1] / (h - 1) - 1
        
        # Warp the moving image
        warped_image = F.grid_sample(moving_image, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped_image

#================================================================================
# 5. MAIN VOXELMORPH MODEL (Refactored for Simplicity)
#================================================================================
class VoxelMorph2d(nn.Module):
    def __init__(self, in_channels):
        super(VoxelMorph2d, self).__init__()
        # The UNet will output a 2-channel deformation field (dx, dy)
        self.unet = UNet(in_channels, 2)
        self.spatial_transform = SpatialTransformation()
        
        # Move the entire model to the correct device
        self.to(device)

    def forward(self, moving_image, fixed_image):
        # --- FIX: All tensors are [B, C, H, W] ---
        # Concatenate along the channel dimension (dim=1)
        x = torch.cat([moving_image, fixed_image], dim=1)
        
        # UNet generates the deformation field
        deformation_field = self.unet(x)
        
        # Apply the spatial transformation
        registered_image = self.spatial_transform(moving_image, deformation_field)
        
        return registered_image, deformation_field

#================================================================================
# 6. LOSS FUNCTIONS (Cleaned Up)
#================================================================================

# In voxelmorph2d.py

def jacobian_determinant(deformation_field):
    """
    Calculates the Jacobian determinant of a 2D deformation field.
    A bug in the previous version's slicing has been corrected here.

    deformation_field: (B, 2, H, W) tensor, where C=0 is dx and C=1 is dy.
    """
    # Extract the x and y displacement fields. Shape: [B, H, W]
    dx = deformation_field[:, 0, :, :]
    dy = deformation_field[:, 1, :, :]

    # Calculate gradients using central differences.
    # The slicing now correctly uses 3 indices for the 3D tensors.
    
    # Gradient in x-direction (change in width)
    # We crop the height dimension to match shapes after calculating gradients.
    J_dx_dx = dx[:, 1:-1, 2:] - dx[:, 1:-1, :-2]
    J_dy_dx = dy[:, 1:-1, 2:] - dy[:, 1:-1, :-2]

    # Gradient in y-direction (change in height)
    # We crop the width dimension to match shapes.
    J_dx_dy = dx[:, 2:, 1:-1] - dx[:, :-2, 1:-1]
    J_dy_dy = dy[:, 2:, 1:-1] - dy[:, :-2, 1:-1]

    # The central difference is over a 2-pixel span, so we divide by 2.
    J_dx_dx /= 2.0
    J_dy_dx /= 2.0
    J_dx_dy /= 2.0
    J_dy_dy /= 2.0
    
    # The Jacobian of the transformation T(p) = p + u(p) is J = I + Du
    # determinant is (1 + dux/dx)(1 + duy/dy) - (dux/dy)(duy/dx)
    determinant = (1.0 + J_dx_dx) * (1.0 + J_dy_dy) - J_dx_dy * J_dy_dx
    
    return determinant
    
    return determinant


def jacobian_regularization_loss(deformation_field):
    """
    惩罚雅可比行列式偏离1的情况
    """
    det = jacobian_determinant(deformation_field)
    
    # 惩罚偏离1的平方误差
    loss = torch.mean((det - 1.0)**2)
    
    # 也可以额外惩罚负的行列式，防止拓扑翻转
    neg_det_loss = torch.sum(F.relu(-det))
    
    return loss + neg_det_loss        

def normalized_cross_correlation(I, J, n=9, eps=1e-5):
    # Assumes I, J are [B, C, H, W] and float tensors
    batch_size, channels, height, width = I.shape
    
    # Define a sum filter for local statistics
    sum_filter = torch.ones((channels, 1, n, n), device=I.device)
    padding = (n - 1) // 2

    I2 = I * I
    J2 = J * J
    IJ = I * J

    # Use grouped convolution for efficient local sum
    I_sum = F.conv2d(I, sum_filter, padding=padding, groups=channels)
    J_sum = F.conv2d(J, sum_filter, padding=padding, groups=channels)
    I2_sum = F.conv2d(I2, sum_filter, padding=padding, groups=channels)
    J2_sum = F.conv2d(J2, sum_filter, padding=padding, groups=channels)
    IJ_sum = F.conv2d(IJ, sum_filter, padding=padding, groups=channels)

    win_size = float(n**2)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - u_I * I_sum * 2 + u_I * u_I * win_size
    J_var = J2_sum - u_J * J_sum * 2 + u_J * u_J * win_size

    cc = (cross * cross) / (I_var * J_var + eps)
    
    # Return 1 - mean NCC for minimization
    return 1 - torch.mean(cc)

def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

def gradient_loss(d_field):
    # Assumes d_field is [B, 2, H, W]
    dy = torch.abs(d_field[:, :, 1:, :] - d_field[:, :, :-1, :])
    dx = torch.abs(d_field[:, :, :, 1:] - d_field[:, :, :, :-1])
    return torch.mean(dx**2) + torch.mean(dy**2)

def dice_loss(pred, target, eps=1e-5):
    # Assumes pred/target are probabilities [0,1]
    intersection = 2 * torch.sum(pred * target, dim=[1, 2, 3])
    union = torch.sum(pred, dim=[1, 2, 3]) + torch.sum(target, dim=[1, 2, 3])
    dice_score = torch.mean((intersection + eps) / (union + eps))
    return 1 - dice_score

def combined_loss(registered_image, fixed_image, deformation_field, n=9, lambda_reg=0.01, alpha=0.5, lambda_jacobian=0.8):
    # Similarity losses
    ncc = normalized_cross_correlation(registered_image, fixed_image, n=n)
    #mse = mse_loss(registered_image, fixed_image)
    dice = dice_loss(registered_image, fixed_image) # Assumes images are normalized to be like segmentation masks
    
    # Regularization loss
    reg_loss = gradient_loss(deformation_field)

    # Jacobian loss
    jacobian_loss = jacobian_regularization_loss(deformation_field)
    
    # Combine losses
    similarity_loss = (1 - alpha) * ncc + alpha * dice
    #similarity_loss = (1 - alpha) * (ncc + mse) + alpha * dice
    total_loss = similarity_loss + lambda_reg * reg_loss + lambda_jacobian * jacobian_loss
    
    return total_loss
