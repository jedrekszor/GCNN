import torch
import numpy as np
import os
import copy
from src.env_variables import BATCH_SIZE, MARK, EPOCHS, MODEL_PATH
from src.model import GCN
import matplotlib.pyplot as plt
from src.resources import validate, accuracy
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)

dataset = MNISTSuperpixels(root="./mnist")
dataset = dataset.shuffle()

train_dataset = dataset[:54000]
val_dataset = dataset[54000:]

loader_training = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
loader_validation = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)


def train(epochs, device, learning_rate=1e-2):
    cnn = GCN(64).to(device, non_blocking=True)
    ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    cnn.train()
    batch_num = len(loader_training)
    min_valid_loss = np.Inf
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        counter = 0
        for batch in loader_training:
            if (counter % 250 == 0):
                print("Epoch ", epoch + 1, "/", epochs, ", Batch ", counter + 1, "/", batch_num, " (",
                  "{:.2f}".format(100 * (counter + 1) / batch_num), "%)")
            counter += 1
            batch.to(device, non_blocking=True)

            pred = cnn(batch.x, batch.edge_index, batch.batch)
            loss = ce(pred, batch.y)

            train_acc += accuracy(pred, batch.y)
            train_loss += float(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_train_acc = train_acc / len(loader_training)
        avg_train_loss = train_loss / len(loader_training)

        print("Validating...")
        avg_valid_acc, avg_valid_loss = validate(cnn, loader_validation, device, ce)
        training_losses.append(float(avg_train_loss))
        validation_losses.append(float(avg_valid_loss))

        if avg_valid_loss < min_valid_loss:
            best_model = copy.deepcopy(cnn)
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            torch.save(best_model.state_dict(),
                       "{}_{}_acc_{:.2f}_loss_{:.5f}".format(MODEL_PATH + "/" + MARK, epoch, avg_valid_acc,
                                                             avg_valid_loss))
            min_valid_loss = avg_valid_loss
        print("Training: accuracy: {:.2f}%, loss: {:.5f}".format(avg_train_acc * 100, avg_train_loss))
        print("Validation: accuracy: {:.2f}%, loss: {:.5f}".format(avg_valid_acc * 100, avg_valid_loss))

        plt.plot(training_losses, label='training')
        plt.plot(validation_losses, label='validation')
        plt.legend()
        fig = plt.gcf()
        plt.show()
        if not os.path.isdir(MODEL_PATH + "/plots"):
            os.makedirs(MODEL_PATH + "/plots")
        fig.savefig(MODEL_PATH + "/plots/plot{}.png".format(epoch + 1))


train(EPOCHS, device)
print("Training finished")
