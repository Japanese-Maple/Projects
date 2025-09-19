import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle as pkl


def get_data_loaders(poisson_dataset_path, batch_size=32, split_ratios=(0.7, 0.2, 0.1)):

    x, y = torch.load(poisson_dataset_path)

    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    x_standardized =  (x - x_mean) / (x_std + 1e-8)
    y_standardized =  (y - y_mean) / (y_std + 1e-8)

    standardization_params = {'x_mean': x_mean, 'x_std': x_std, 
                              'y_mean': y_mean, 'y_std': y_std}
    
    with open("scaler.pkl", "wb") as f:
        pkl.dump(standardization_params, f)

    dataset = TensorDataset(x_standardized, y_standardized)
    
    # train / val / test
    total_size = len(dataset)
    train_size = int(split_ratios[0] * total_size)
    val_size   = int(split_ratios[1] * total_size)
    test_size  = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=9, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, num_workers=9)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, num_workers=9)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    path = 'poisson_dataset.pt'
    train, validate, test = get_data_loaders(path, batch_size=256)

    