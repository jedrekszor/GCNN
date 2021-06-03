import torch
import numpy as np
import os
import inspect
import sys
from env_variables import TRAINING_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE, IMG_SIZE, MARK, EPOCHS, MODEL_PATH
from model import Net
import matplotlib.pyplot as plt
import copy
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from resources import validate, data_transforms, accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)

set_training = ImageFolder(TRAINING_DIR, transform=data_transforms(TRAINING_DIR))
set_validation = ImageFolder(VALIDATION_DIR, transform=data_transforms(VALIDATION_DIR))
set_test = ImageFolder(TEST_DIR, transform=data_transforms(TEST_DIR))

loader_training = DataLoader(set_training, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_validation = DataLoader(set_validation, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_test = DataLoader(set_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


def train(epochs, device, learning_rate=1e-3):
    cnn = Net().to(device)
    ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    batch_num = len(loader_training)
    min_valid_loss = np.Inf
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0

        for i, (images, labels) in enumerate(loader_training):
            print("Epoch ", epoch + 1, "/", epochs, ", Batch ", i + 1, "/", batch_num, " (",
                  "{:.2f}".format(100 * (i + 1) / batch_num), "%)")

            images = images.to(device)
            labels = labels.to(device)

            pred = cnn(images)
            loss = ce(pred, labels)

            train_acc += accuracy(pred, labels)
            train_loss += float(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_acc = train_acc / len(loader_training)
        avg_train_loss = train_loss / len(loader_training)

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
        print("Training: accuracy: {:.2f}%, loss: {:.5f}".format(avg_train_acc*100, avg_train_loss))
        print("Validation: accuracy: {:.2f}%, loss: {:.5f}".format(avg_valid_acc*100, avg_valid_loss))

        plt.plot(training_losses, label='training')
        plt.plot(validation_losses, label='validation')
        plt.legend()
        fig = plt.gcf()
        plt.show()
        if not os.path.isdir(MODEL_PATH + "/plots"):
            os.makedirs(MODEL_PATH + "/plots")
        fig.savefig(MODEL_PATH + "/plots/plot{}.png".format(epoch+1))


train(EPOCHS, device)
print("Training finished")
