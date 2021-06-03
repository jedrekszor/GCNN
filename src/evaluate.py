import torch
import numpy as np
import os
import inspect
import sys
from env_variables import TRAINING_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE, MODEL_PATH
from model import Net
import matplotlib.pyplot as plt
import copy
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from resources import validate, data_transforms, accuracy, save_wrong, plot_confusion_matrix

MODEL_LOAD = "../models/resnet18/resnet18_24_acc_0.91_loss_0.24456"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)

set_training = ImageFolder(TRAINING_DIR, transform=data_transforms(TRAINING_DIR))
set_validation = ImageFolder(VALIDATION_DIR, transform=data_transforms(VALIDATION_DIR))
set_test = ImageFolder(TEST_DIR, transform=data_transforms(TEST_DIR))

loader_training = DataLoader(set_training, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_validation = DataLoader(set_validation, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_test = DataLoader(set_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


def evaluate():
    cnn = Net().to(device)
    cnn.load_state_dict(torch.load(MODEL_LOAD))
    ce = torch.nn.CrossEntropyLoss()
    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    train_loss = 0.0
    train_acc = 0.0
    batch_num = len(loader_training)
    train_pred = []
    train_true = []

    if not os.path.isdir(MODEL_PATH + "/wrong/train"):
        os.makedirs(MODEL_PATH + "/wrong/train")
    if not os.path.isdir(MODEL_PATH + "/wrong/val"):
        os.makedirs(MODEL_PATH + "/wrong/val")

    for i, (images, labels) in enumerate(loader_training):
        print("Batch ", i + 1, "/", batch_num, " (",
              "{:.2f}".format(100 * (i + 1) / batch_num), "%)")
        images = images.to(device)
        labels = labels.to(device)

        x = cnn(images)
        loss = ce(x, labels)

        value, pred = torch.max(x, 1)
        for j, k in enumerate(images):
            if pred[j] != labels[j]:
                save_wrong((i + j), images[j], set_training.classes[pred[j]], set_training.classes[labels[j]], MODEL_PATH, "train")
        train_acc += accuracy(x, labels)
        train_loss += float(loss)
        pred = pred.data.cpu()
        labels = labels.data.cpu()
        train_pred.extend(list(pred.numpy()))
        train_true.extend(list(labels.numpy()))
    avg_train_acc = train_acc / len(loader_training)
    avg_train_loss = train_loss / len(loader_training)

    print("###########################")

    val_loss = 0.0
    val_acc = 0.0
    batch_num = len(loader_validation)
    val_pred = []
    val_true = []
    for i, (images, labels) in enumerate(loader_validation):
        print("Batch ", i + 1, "/", batch_num, " (",
              "{:.2f}".format(100 * (i + 1) / batch_num), "%)")
        images = images.to(device)
        labels = labels.to(device)

        x = cnn(images)
        loss = ce(x, labels)

        value, pred = torch.max(x, 1)
        for j, k in enumerate(images):
            if pred[j] != labels[j]:
                save_wrong((i + j), images[j], set_validation.classes[pred[j]], set_validation.classes[labels[j]], MODEL_PATH, "val")
        val_acc += accuracy(x, labels)
        val_loss += float(loss)
        pred = pred.data.cpu()
        labels = labels.data.cpu()
        val_pred.extend(list(pred.numpy()))
        val_true.extend(list(labels.numpy()))
    avg_val_acc = val_acc / len(loader_validation)
    avg_val_loss = val_loss / len(loader_validation)

    print("########## TRAINING ##########")
    print("Accuracy: {:.2f}%, loss: {:.5f}".format(avg_train_acc * 100, avg_train_loss))
    cm = confusion_matrix(train_true, train_pred)
    plot_confusion_matrix(cm, set_training.classes, "Training")
    fig = plt.gcf()
    plt.show()
    fig.savefig(MODEL_PATH + "/CM_train.png")

    print("########## VALIDATION ##########")
    print("Accuracy: {:.2f}%, loss: {:.5f}".format(avg_val_acc * 100, avg_val_loss))
    cm = confusion_matrix(val_true, val_pred)
    plot_confusion_matrix(cm, set_validation.classes, "Validation")
    fig = plt.gcf()
    plt.show()
    fig.savefig(MODEL_PATH + "/CM_val.png")


evaluate()
