import torch
import numpy as np

# Class for data normalization (standardization)
class StandardScaler:
    def __init__(self, data):
        self.mean = data.mean()
        self.std = data.std()

    # Standardize the data
    def transform(self, data):
        return (data - self.mean) / self.std

    # Reverse the standardization process
    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(
                data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(
                data.device).type(data.dtype)
        return (data * self.std) + self.mean

# Function to create input-output pairs for the given horizon
def add_horizon(data, horizon):
    X = []
    Y = []
    for i in range(len(data) - 2 * horizon + 1):
        X.append(data[i:i+horizon])
        Y.append(data[i+horizon:i+(2 * horizon)])
    return np.array(X), np.array(Y)

# Function to convert numpy arrays to PyTorch dataloaders
def torch_loader(X, Y, batch_size, shuffle, drop_last):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

# Function to prepare the dataset and split it into train, validation, and test sets
def get_dataset(config):
    # Load data from the file and extract the required data
    data = np.load(config["dataset_path"])['data'][:, :, 0]
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    
    # Create a scaler object and normalize the data
    scaler = StandardScaler(data)
    data = scaler.transform(data)
    
    # Split the data into train, validation, and test sets
    data_len = data.shape[0]
    test_data = data[-int(data_len * config["test_ratio"]):]
    val_data = data[-int(data_len * (config["test_ratio"] + config["val_ratio"])):-int(data_len * config["test_ratio"])]
    train_data = data[:-int(data_len * (config["test_ratio"] + config["val_ratio"]))]
    
    # Create input-output pairs for each dataset
    x_train, y_train = add_horizon(train_data, config["horizon"])
    x_val, y_val = add_horizon(val_data, config["horizon"])
    x_test, y_test = add_horizon(test_data, config["horizon"])
    
    # Convert numpy arrays to PyTorch dataloaders
    train_dataloader = torch_loader(
        x_train, y_train, config["batch_size"], shuffle=True, drop_last=True)
    val_dataloader = torch_loader(
        x_val, y_val, config["batch_size"], shuffle=False, drop_last=True)
    test_dataloader = torch_loader(
        x_test, y_test, config["batch_size"], shuffle=False, drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader, scaler
