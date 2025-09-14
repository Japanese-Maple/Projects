import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from PIL import Image
import shutil
#-----------------------------

# ----- 1. Generate data -----
l = 3
torch.manual_seed(0)
x = torch.linspace(-l, l, 100).unsqueeze(1)  # shape (N, 1)
y_true = torch.relu(x)      
y = y_true + torch.randn_like(y_true)/5  # add noise

# ----- 2. Model -----
# We use Fourier Series as a base and approximate the coefficients via GD.
class FS_Fit(nn.Module):
    def __init__(self, x, n, a):
        super().__init__()
        self.n = n
        self.a = a
        self.x = x
        self.cos = torch.cat([torch.cos((torch.pi*i*x)/self.a) for i in reversed(range(self.n+1))], dim=1)
        self.sin = torch.cat([torch.sin((torch.pi*i*x)/self.a) for i in reversed(range(self.n+1))], dim=1)
        # learnable coefficients
        self.coeffs_sin = nn.Parameter(torch.ones(n+1, 1)*3)
        self.coeffs_cos = nn.Parameter(torch.ones(n+1, 1)*3)
    
    def forward(self, x):
        # Build the fourier series       
        sin = self.sin @ self.coeffs_sin        
        cos = self.cos @ self.coeffs_cos
        return sin + cos

# ----- 3. Training -----
n = 11
a = 3.1
model = FS_Fit(x, n, a)
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

output_dir = "FS_fit"
os.makedirs(output_dir, exist_ok=True)

for epoch in range(300):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        frame_idx = epoch // 10
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)  
        ax.scatter(x.numpy(), y.numpy(), alpha=0.5, label="Data")
        ax.plot(x.numpy(), y_pred.detach().numpy(), color="red", label=f"$\\mathcal{{F}}$ $(n={n}, a={a})$")
        ax.plot(x.numpy(), y_true.numpy(), "g--", label="True $f(x)$")
        ax.set_xlim(-l, l)
        ax.set_ylim(min(y)-0.1, max(y)+0.1)
        # ax.set_aspect('equal') 
        ax.legend()
        fig.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
        plt.close(fig)

# ----- 4. Video -----
sample_frame = Image.open(f"{output_dir}/frame_0000.png")
width, height = sample_frame.size
even_width = width if width % 2 == 0 else width + 1
even_height = height if height % 2 == 0 else height + 1
print(f"Sample frame resolution: {width}x{height} → using {even_width}x{even_height} for video")

video_filename = "FS_fit.mp4"
print("Stitching frames into video...")

ffmpeg_cmd = f"""
ffmpeg -loglevel error -y -framerate 10 -i {output_dir}/frame_%04d.png \
-vf "scale={even_width}:{even_height}:flags=lanczos" \
-c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p {video_filename}
"""
os.system(ffmpeg_cmd)

# ----- 5. Clean up -----
print("Deleting frames folder...")
shutil.rmtree(output_dir)

print(f"Video saved to {video_filename}")
print("Learned coefficients:")
print("cos:", model.coeffs_cos.detach().view(-1).numpy())
print("sin:", model.coeffs_sin.detach().view(-1).numpy())
