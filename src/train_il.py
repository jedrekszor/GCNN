import torch
import numpy as np
import os
import copy
from src.env_variables import BATCH_SIZE, MARK, EPOCHS, MODEL_PATH
from src.model import GCN
import matplotlib.pyplot as plt
from src.functions import validate, accuracy
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader

def train(model, loader_training, loader_validation, epochs, device, learning_rate, loss_function):

    model = model.to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    model.train()
    batch_number = len(loader_training)
    min_valid_loss = np.Inf
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        counter = 0
        for batch in loader_training:
            if (counter % 250 == 0):
                print("Epoch ", epoch + 1, "/", epochs, ", Batch ", counter + 1, "/", batch_number, " (",
                  "{:.2f}".format(100 * (counter + 1) / batch_number), "%)")
            counter += 1

            batch.to(device, non_blocking=True)

            prediction = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_function(prediction, batch.y)

            train_accuracy += accuracy(prediction, batch.y)
            train_loss += float(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_accuracy = train_accuracy / len(loader_training)
        avg_train_loss = train_loss / len(loader_training)

        print("Validating...")
        avg_valid_acc, avg_valid_loss = validate(model, loader_validation, device, loss_function)
        training_losses.append(float(avg_train_loss))
        validation_losses.append(float(avg_valid_loss))

        if avg_valid_loss < min_valid_loss:
            best_model = copy.deepcopy(model)
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            torch.save(best_model.state_dict(),
                       "{}_{}_acc_{:.2f}_loss_{:.5f}".format(MODEL_PATH + "/" + MARK, epoch, avg_valid_acc,
                                                             avg_valid_loss))
            min_valid_loss = avg_valid_loss
        print("Training: accuracy: {:.2f}%, loss: {:.5f}".format(avg_train_accuracy * 100, avg_train_loss))
        print("Validation: accuracy: {:.2f}%, loss: {:.5f}".format(avg_valid_acc * 100, avg_valid_loss))

        plt.plot(training_losses, label='training')
        plt.plot(validation_losses, label='validation')
        plt.legend()
        figure = plt.gcf()
        plt.show()
        if not os.path.isdir(MODEL_PATH + "/plots"):
            os.makedirs(MODEL_PATH + "/plots")
        figure.savefig(MODEL_PATH + "/plots/plot{}.png".format(epoch + 1))

    print("Training finished")
