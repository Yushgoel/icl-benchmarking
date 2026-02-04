
import numpy as np
import torch


def load_data(data_path):
    data = np.load(data_path)
    return (
        torch.from_numpy(data['X_train']).float(),
        torch.from_numpy(data['y_train']).float(),
        torch.from_numpy(data['X_test']).float(),
        torch.from_numpy(data['y_test']).float()
    )
