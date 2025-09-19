import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = self.skip(x)
        x = self.dropout(self.act1(self.bn1(self.conv1(x))))
        x = self.dropout(self.bn2(self.conv2(x)))
        return nn.GELU()(x + residual)

class DeepCNN(nn.Module):
    def __init__(self, input_channels=1, base_channels=16, num_blocks=4, num_classes=1):
        super().__init__()
        self.input_proj = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[
            ResidualBlock(base_channels, base_channels) for _ in range(num_blocks)
        ])
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.out(x)
    
# --------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    model = DeepCNN(
        input_channels=1,
        base_channels=16,
        num_blocks=4,  
        num_classes=1
    )
        
