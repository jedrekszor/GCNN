import torch
import os
from src.env_variables import MODEL_PATH
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.functions import accuracy, plot_confusion_matrix


def evaluate(model, loader_training, loader_test, device, loss_function, num_classes):
    model = model.to(device, non_blocking=True)

    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    train_loss = 0.0
    train_acc = 0.0
    batch_number = len(loader_training)
    train_pred = []
    train_true = []

    counter = 0
    for batch in loader_training:
        if (counter % 250 == 0):
            print("Batch ", counter + 1, "/", batch_number, " (",
                  "{:.2f}".format(100 * (counter + 1) / batch_number), "%)")
        counter += 1

        batch.to(device, non_blocking=True)
        prediction = model(batch)
        loss = loss_function(prediction, batch.y)

        value, pred = torch.max(prediction, 1)
        train_acc += accuracy(prediction, batch.y)
        train_loss += float(loss)

        pred = pred.data.cpu()
        labels = batch.y.data.cpu()
        train_pred.extend(list(pred.numpy()))
        train_true.extend(list(labels.numpy()))
    avg_train_acc = train_acc / len(loader_training)
    avg_train_loss = train_loss / len(loader_training)

    print("###########################")

    test_loss = 0.0
    test_acc = 0.0
    batch_number = len(loader_test)
    test_pred = []
    test_true = []
    counter = 0
    for batch in loader_test:
        if (counter % 250 == 0):
            print("Batch ", counter + 1, "/", batch_number, " (",
                  "{:.2f}".format(100 * (counter + 1) / batch_number), "%)")
        counter += 1

        batch.to(device, non_blocking=True)
        prediction = model(batch)
        loss = loss_function(prediction, batch.y)

        value, pred = torch.max(prediction, 1)
        test_acc += accuracy(prediction, batch.y)
        test_loss += float(loss)

        pred = pred.data.cpu()
        labels = batch.y.data.cpu()
        test_pred.extend(list(pred.numpy()))
        test_true.extend(list(labels.numpy()))
    avg_test_acc = test_acc / len(loader_test)
    avg_test_loss = test_loss / len(loader_test)

    print("########## TRAINING ##########")
    print("Accuracy: {:.2f}%, loss: {:.5f}".format(avg_train_acc * 100, avg_train_loss))
    cm = confusion_matrix(train_true, train_pred)
    plot_confusion_matrix(cm, num_classes, "Training")
    fig = plt.gcf()
    plt.show()
    fig.savefig(MODEL_PATH + "/CM_train.png")

    print("########## TEST ##########")
    print("Accuracy: {:.2f}%, loss: {:.5f}".format(avg_test_acc * 100, avg_test_loss))
    cm = confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(cm, num_classes, "Test")
    fig = plt.gcf()
    plt.show()
    fig.savefig(MODEL_PATH + "/CM_test.png")
