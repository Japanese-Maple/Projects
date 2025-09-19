import torch
import matplotlib.pyplot as plt
from FNO import FNO

# Load full dataset
inputs, targets = torch.load("poisson_dataset_train.pt")

# Choose one sample
idx = 42  
input_sample = inputs[idx].unsqueeze(0)   # [1, 1, H, W]
target_sample = targets[idx].unsqueeze(0) # [1, 1, H, W]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_sample = input_sample.to(device)
target_sample = target_sample.to(device)

# Permute to [B, H, W, C] for FNO
input_sample = input_sample.permute(0, 2, 3, 1)
target_sample = target_sample.permute(0, 2, 3, 1)

# Load pre-trained model
model = FNO(modes1=16, modes2=16, width=32)
model.load_state_dict(torch.load("model_FNO_GELU.pt"))
model.eval()
model.to(device)

# Predict
with torch.no_grad():
    pred = model(input_sample)  # [1, H, W, C]

# Move data to CPU for plotting
target_img = target_sample.squeeze().cpu().numpy()
pred_img = pred.squeeze().cpu().numpy()

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(target_img, cmap='viridis')
axes[0].set_title("Target")

axes[1].imshow(pred_img, cmap='viridis')
axes[1].set_title("Prediction")

error_map = abs(target_img - pred_img)
im = axes[2].imshow(error_map, cmap='viridis')
axes[2].set_title("Absolute Error")

# Add colorbars
for ax in axes:
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()