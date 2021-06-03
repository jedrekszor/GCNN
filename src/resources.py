import torch
from torchvision import transforms as T
from env_variables import BATCH_SIZE, TRAINING_DIR, VALIDATION_DIR, TEST_DIR, IMG_SIZE
import itertools
import numpy as np
import matplotlib.pyplot as plt


def data_transforms(phase=None):
    if phase == TRAINING_DIR:
        data_T = T.Compose([
            # T.Grayscale(),
            T.Resize(size=(256, 256)),
            T.RandomRotation(degrees=(-20, +20)),
            T.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
            T.ToTensor()
        ])

    elif phase == VALIDATION_DIR or phase == TEST_DIR:
        data_T = T.Compose([
            # T.Grayscale(),
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
            T.ToTensor()
        ])

    return data_T


def validate(model, data, device, ce):  # accuracy
    valid_acc = 0.0
    valid_loss = 0.0

    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = ce(pred, labels)

        valid_acc += accuracy(pred, labels)
        valid_loss += float(loss)

    avg_valid_acc = valid_acc / len(data)
    avg_valid_loss = valid_loss / len(data)
    return avg_valid_acc, avg_valid_loss


def accuracy(y_pred, y_true):
    y_pred = torch.exp(y_pred)
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def save_wrong(id, image, pred, true, path, set):
    img = T.functional.to_pil_image(image)
    img.save(path + "/wrong/" + set + "/" + "{}_pred_{}_actual_{}.png".format(
        id, pred, true))


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
