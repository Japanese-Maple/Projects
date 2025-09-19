import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from FNO import FNO
from Data_Loader import get_data_loaders
import numpy as np
from pytorch_msssim import ssim
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, test_loader, lr=0.001, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")


    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optimizer,
                                                            mode='min',             # Use 'min' for loss; use 'max' for metrics like accuracy
                                                            factor=0.5,             # Reduce LR by this factor
                                                            patience=2,             # Wait for 5 epochs before reducing LR
                                                            verbose=True            # Print updates to console
                                                        )

    loss_fn = nn.MSELoss()

    # def relative_l2_loss(pred, true):
    #     return torch.norm(pred - true) / torch.norm(true)

    # loss_fn = relative_l2_loss   

    # def poisson_loss(pred, true):
    #     """
    #     Complete loss function for Poisson equation training with boundary conditions
        
    #     Args:
    #         pred: Model predictions (B, H, W, 1)
    #         true: Ground truth solutions (B, H, W, 1)
        
    #     Returns:
    #         Combined loss value
    #     """
    #     # Convert to (B, 1, H, W) format for easier boundary access
    #     pred_tensor = pred.permute(0, 3, 1, 2)  # (B, 1, H, W)
    #     true_tensor = true.permute(0, 3, 1, 2)  # (B, 1, H, W)
        
    #     # 1. Main MSE loss between prediction and ground truth
    #     mse_loss = F.mse_loss(pred, true)
        
    #     # 2. Boundary condition loss - enforce zero Dirichlet boundary conditions
    #     boundary_loss = (
    #         # Top boundary (y = 0)
    #         F.mse_loss(pred_tensor[:, :, 0, :], torch.zeros_like(pred_tensor[:, :, 0, :])) +
    #         # Bottom boundary (y = H)
    #         F.mse_loss(pred_tensor[:, :, -1, :], torch.zeros_like(pred_tensor[:, :, -1, :])) +
    #         # Left boundary (x = 0)
    #         F.mse_loss(pred_tensor[:, :, :, 0], torch.zeros_like(pred_tensor[:, :, :, 0])) +
    #         # Right boundary (x = H)
    #         F.mse_loss(pred_tensor[:, :, :, -1], torch.zeros_like(pred_tensor[:, :, :, -1]))
    #     )
        
    #     # 3. Physics-informed loss - check if prediction satisfies Poisson equation
    #     def laplacian_loss(u):
    #         """Compute discrete Laplacian for physics loss"""
    #         # u shape: (B, 1, H, W)
    #         # Discrete Laplacian using finite differences
    #         u_xx = u[:, :, 2:, 1:-1] - 2*u[:, :, 1:-1, 1:-1] + u[:, :, :-2, 1:-1]
    #         u_yy = u[:, :, 1:-1, 2:] - 2*u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2]
    #         laplacian = u_xx + u_yy
    #         return laplacian
        
        
    #     # Extract source term from the relationship between true solution and its Laplacian
    #     pred_laplacian = laplacian_loss(pred_tensor)
    #     true_laplacian = laplacian_loss(true_tensor)
        
    #     # Physics loss - predicted Laplacian should match true Laplacian
    #     physics_loss = F.mse_loss(pred_laplacian, true_laplacian)
        
    #     # 4. Smoothness regularization - prevent oscillations
    #     def total_variation_loss(u):
    #         """Total variation regularization for smoothness"""
    #         # u shape: (B, 1, H, W)
    #         tv_h = torch.mean(torch.abs(u[:, :, 1:, :] - u[:, :, :-1, :]))
    #         tv_w = torch.mean(torch.abs(u[:, :, :, 1:] - u[:, :, :, :-1]))
    #         return tv_h + tv_w
        
    #     smoothness_loss = total_variation_loss(pred_tensor)
        
    #     # 5. Combine all losses with appropriate weights
    #     total_loss = (
    #         1.0 * mse_loss +           # Main reconstruction loss
    #         0.1 * boundary_loss +      # Boundary condition enforcement
    #         0.05 * physics_loss +      # Physics-informed constraint
    #         0.01 * smoothness_loss     # Smoothness regularization
    #     )
        
    #     return total_loss 

    # def magnitude_aware_loss(pred, target):
    #     mse_loss = F.mse_loss(pred, target)
    #     # Add magnitude consistency penalty
    #     pred_range = pred.max() - pred.min()
    #     target_range = target.max() - target.min()
    #     magnitude_penalty = torch.log(pred_range / (target_range + 1e-8))
    #     return mse_loss + 0.001 * magnitude_penalty
    
    # loss_fn = magnitude_aware_loss

    train_loss_values = []
    validation_loss_values = []
    
    for epoch in tqdm(range(epochs + 1), desc="Training"):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            X_batch = X_batch.permute(0, 2, 3, 1)  # (B, H, W, 1)
            Y_batch = Y_batch.permute(0, 2, 3, 1)  # (B, H, W, 1)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # scheduler.step()  
        avg_train_loss = train_loss / len(train_loader)
        train_loss_values.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                X_val = X_val.permute(0, 2, 3, 1)  # (B, H, W, 1)
                Y_val = Y_val.permute(0, 2, 3, 1)  # (B, H, W, 1)
                pred_val = model(X_val)
                val_loss += loss_fn(pred_val, Y_val).item()
        avg_val_loss = val_loss / len(val_loader)
        validation_loss_values.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if epoch % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.7f}, Validation Loss = {avg_val_loss:.7f}")

    torch.save(model.state_dict(), "model_FNO_noise.pt")
    print("✅ Model saved")
    np.save('train_loss_FNO.npy', train_loss_values)
    np.save('val_loss_FNO.npy',   validation_loss_values)

# Run training
if __name__ == "__main__":  
    torch.manual_seed(33)
    model = FNO(modes1=16, modes2=16, width=64)
    epochs = 30
    path = "poisson_dataset_train_noise.pt"
    train_loader, val_loader, test_loader = get_data_loaders(path, batch_size=64)    
    train_model(model, train_loader, val_loader, test_loader, lr=0.001, epochs=epochs)
   


