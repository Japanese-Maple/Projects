import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from CNN import DeepCNN
from Data_Loader import get_data_loaders
import numpy as np

def train_model(model, train_loader, val_loader, test_loader, lr=0.001, epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optimizer,
                                                            mode='min',             # Use 'min' for loss; use 'max' for metrics like accuracy
                                                            factor=0.5,             # Reduce LR by this factor
                                                            patience=2,             # Wait for 5 epochs before reducing LR
                                                            verbose=True            # Print updates to console
                                                        )
    loss_fn = nn.MSELoss()

    train_loss_values = []
    validation_loss_values = []
    
    for epoch in tqdm(range(epochs + 1)):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)            
            loss.backward()
            optimizer.step()            
            optimizer.zero_grad()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_loss_values.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                pred_val = model(X_val)
                val_loss += loss_fn(pred_val, Y_val).item()
        avg_val_loss = val_loss / len(val_loader)
        validation_loss_values.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if epoch % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.7f}, Validation Loss = {avg_val_loss:.7f}")

    torch.save(model.state_dict(), "model_CNN.pt")
    print("✅ Model saved")
    np.save('train_loss_CNN.npy', train_loss_values)
    np.save('val_loss_CNN.npy',   validation_loss_values)

# Run training
if __name__ == "__main__":
    model = DeepCNN(input_channels=1, base_channels=32, num_blocks=4, num_classes=1)
    path = "poisson_dataset_train.pt"
    train_loader, val_loader, test_loader = get_data_loaders(path, batch_size=64)
    train_model(model, train_loader, val_loader, test_loader, lr=0.001, epochs=100)
