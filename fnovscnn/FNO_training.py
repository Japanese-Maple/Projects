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
    print("Model saved")
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
   


