import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. [FFT --> linear transform --> Inverse FFT]   
    """
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()        

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2]  = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, channel_input=1, output_channel=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width  # fixed internally

        self.padding = 18
        self.fc0 = nn.Linear(channel_input + 2, self.width)  # +2 for (x, y)

        self.spec0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.spec1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.spec2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.spec3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0 = nn.Conv2d(self.width, self.width, 1)
        self.conv1 = nn.Conv2d(self.width, self.width, 1)
        self.conv2 = nn.Conv2d(self.width, self.width, 1)
        self.conv3 = nn.Conv2d(self.width, self.width, 1)

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_channel)

    def forward(self, x):
        dropout = False

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        # x = F.pad(x, [0,self.padding, 0,self.padding])
        
        if dropout == True:
            x1 = self.spec0(x) + self.conv0(x)
            x1 = self.dropout(F.gelu(x1))
            x2 = self.spec1(x1) + self.conv1(x1)
            x2 = self.dropout(F.gelu(x2))
            x3 = self.spec2(x2) + self.conv2(x2)
            x3 = self.dropout(F.gelu(x3))
            x4 = self.spec3(x3) + self.conv3(x3)

        else:
            x1 = self.spec0(x) + self.conv0(x)
            x1 = F.gelu(x1)
            x2 = self.spec1(x1) + self.conv1(x1)
            x2 = F.gelu(x2)
            x3 = self.spec2(x2) + self.conv2(x2)
            x3 = F.gelu(x3)
            x4 = self.spec3(x3) + self.conv3(x3)

        # x4 = x4[..., :-self.padding, :-self.padding]
        x = x4.permute(0, 2, 3, 1)  # B H W C
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).unsqueeze(0).repeat(size_y, 1).T
        gridy = torch.linspace(0, 1, size_y, device=device).unsqueeze(0).repeat(size_x, 1)

        gridx = gridx.unsqueeze(0).unsqueeze(-1).repeat(batchsize, 1, 1, 1)
        gridy = gridy.unsqueeze(0).unsqueeze(-1).repeat(batchsize, 1, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)