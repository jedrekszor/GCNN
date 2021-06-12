import torch
from torchvision import transforms as T
import itertools
import numpy as np
import matplotlib.pyplot as plt


def validate(model, data, device, ce):
    valid_acc = 0.0
    valid_loss = 0.0

    for batch in data:
        batch.to(device, non_blocking=True)

        pred = model(batch)
        loss = ce(pred, batch.y)

        valid_acc += accuracy(pred, batch.y)
        valid_loss += float(loss)

    avg_valid_acc = valid_acc / len(data)
    avg_valid_loss = valid_loss / len(data)
    return avg_valid_acc, avg_valid_loss


def validate_t(model, data, device, ce):
    valid_acc = 0.0
    valid_loss = 0.0

    data = data.to(device)

    pred = model(data)[data.test_mask]
    loss = ce(pred, data.y[data.test_mask])

    valid_acc += accuracy(pred, data.y[data.test_mask])
    valid_loss += float(loss)

    return valid_acc, valid_loss


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
