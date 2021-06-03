import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader

from src.env_variables import BATCH_SIZE, MARK, EPOCHS, MODEL_PATH
from src.model import GCN
from src.functions import validate, accuracy
from src.train_li import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)

# ### Experiment 1
dataset_mnist = MNISTSuperpixels(root="../mnist")
dataset_mnist = dataset_mnist.shuffle()

dataset_mnist_train, dataset_mnist_val = dataset_mnist[:54000], dataset_mnist[54000:]

loader_mnist_train = DataLoader(dataset_mnist_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
loader_mnist_val = DataLoader(dataset_mnist_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

model = GCN(64).to(device, non_blocking=True)

train(model, loader_mnist_train, loader_mnist_val, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())

validate(model, load, device, torch.nn.CrossEntropyLoss())


# ### Experiment 2

model = GCN(64).to(device, non_blocking=True)

train(model, loader_mnist_train, loader_mnist_val, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())

validate(model, load, device, torch.nn.CrossEntropyLoss())


# ### Experiment 3
