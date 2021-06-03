from torch import nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)

        self.ln1 = nn.Linear(1000, 256)
        self.ln2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.model(x)
        return self.ln2(self.ln1(x))
