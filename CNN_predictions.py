import torch
import matplotlib.pyplot as plt
from CNN import DeepCNN
import pickle as pkl
import numpy as np

# Load the dataset
inputs, targets = torch.load("poisson_dataset.pt")

model = DeepCNN(
        input_channels=1,
        base_channels=32,
        num_blocks=3,  
        num_classes=1
    )
model.load_state_dict(torch.load("model_CNN.pt"))
model.eval()
model.cuda()

# Choose a sample index
rand_num = torch.randint(1, 501, (1,))
idx = rand_num.item()
print(idx)

input_sample = inputs[idx].unsqueeze(0).cuda()  
target_sample = targets[idx].squeeze()

# Run prediction
with torch.no_grad():
    output = model(input_sample).squeeze().cpu()

# Denormalize
output = output.numpy()
target_sample = target_sample.numpy()
inputs = inputs[idx].squeeze().numpy()

# Plot
extent = [0, 2, 0, 2]  # Adjust based on your actual physical domain (use [0, H, 0, H] if needed)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Poisson Equation Approximation for Two Charges')
# Plot using contourf
cs0 = axs[0].contourf(inputs, levels=30, cmap='viridis', extent=extent, origin="lower")
axs[0].set_title("Input: f(x, y)")
cs1 = axs[1].contourf(target_sample, levels=30, cmap='viridis', extent=extent, origin="lower")
axs[1].set_title("Ground Truth: u(x, y)")
cs2 = axs[2].contourf(output, levels=30, cmap='viridis', extent=extent, origin="lower")
axs[2].set_title("Predicted: û(x, y)")

# Optional: Add colorbars if you like
for ax, cs in zip(axs, [cs0, cs1, cs2]):
    plt.colorbar(cs, ax=ax)

# Clean up axes
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


plt.tight_layout()
plt.show()

# def f(x, y):    
#     return np.sin(x + y) - 3*np.cos(3*x)

# x = np.linspace(0, 2, 92)
# y = np.linspace(0, 2, 92)
# X,Y = np.meshgrid(x,y)

# input_tensor = torch.tensor(f(X, Y), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)  

# input_tensor = input_tensor.to(device)  
# res = model(input_tensor)

# plt.imshow(res.squeeze().detach().cpu().numpy())
# plt.show()