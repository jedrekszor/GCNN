import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from src.env_variables import BATCH_SIZE, MARK, EPOCHS, MODEL_PATH
from src.model import GCN
from src.functions import validate, accuracy
from src.train_il import train
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)

# ### Experiment 1
# dataset_mnist = MNISTSuperpixels(root="../mnist")
# dataset_mnist = dataset_mnist.shuffle()
#
transform = T.Cartesian(cat=False)
dataset_mnist_train, dataset_mnist_val = MNISTSuperpixels("../mnist", True, transform=transform), MNISTSuperpixels("../mnist", False, transform=transform)

loader_mnist_train = DataLoader(dataset_mnist_train, batch_size=64, shuffle=True)
loader_mnist_val = DataLoader(dataset_mnist_val, batch_size=64)

model = GCN(dataset_mnist_train).to(device, non_blocking=True)

train(model, loader_mnist_train, loader_mnist_val, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())

validate(model, loader_mnist_train, loader_mnist_val, device, F.nll_loss())


# ### Experiment 2

model = GCN(dataset_mnist_train).to(device, non_blocking=True)

train(model, loader_mnist_train, loader_mnist_val, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())

validate(model, loader_mnist_train, loader_mnist_val, device, F.nll_loss())


# ### Experiment 3
