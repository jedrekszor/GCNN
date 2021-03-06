import torch
from torch_geometric.datasets import MNISTSuperpixels, Planetoid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from src.env_variables import EPOCHS
from src.model import GCN1, GCN2, GCN_t1, GCN_t2
from src.evaluate import evaluate, evaluate_t
from src.train import train, train_t
from src.visualize import visualize, visualize_t


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used: ", device)

print("Preparing dataset MNIST")
transform = T.Cartesian(cat=False)
dataset_mnist_train, dataset_mnist_val = MNISTSuperpixels("../mnist", True, transform=transform), MNISTSuperpixels("../mnist", False, transform=transform)

loader_mnist_train = DataLoader(dataset_mnist_train, batch_size=64, shuffle=True)
loader_mnist_val = DataLoader(dataset_mnist_val, batch_size=64)

# print("Generating visualization")
# visualize(dataset_mnist)


print("Preparing dataset Planetoid")
transform_t = T.Compose([
    T.AddTrainValTestMask('train_rest', num_val=500, num_test=500),
    T.TargetIndegree(),
])
dataset_planetoid = Planetoid("../planetoid", "Cora", transform=transform_t)
dataset_planetoid_graph = dataset_planetoid[0]

# ### Experiment 1
# model = GCN1(dataset_mnist_train).to(device, non_blocking=True)
# train(model, loader_mnist_train, loader_mnist_val, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())
# evaluate(model, loader_mnist_train, loader_mnist_val, device, torch.nn.CrossEntropyLoss(), dataset_mnist_train.num_classes)


# ### Experiment 2
# model = GCN2(dataset_mnist_train).to(device, non_blocking=True)
# train(model, loader_mnist_train, loader_mnist_val, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())
# evaluate(model, loader_mnist_train, loader_mnist_val, device, torch.nn.CrossEntropyLoss(), dataset_mnist_train.num_classes)


# ### Experiment 3
# model = GCN_t1(dataset_planetoid).to(device, non_blocking=True)
# model.load_state_dict(torch.load("../out/models/e4/e4_13_acc_0.83_loss_0.55229"))
# train_t(model, dataset_planetoid_graph, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())
# evaluate_t(model, dataset_planetoid_graph, device, torch.nn.CrossEntropyLoss(), dataset_planetoid.num_classes)
#
#
# # ### Experiment 4
model = GCN_t2(dataset_planetoid).to(device, non_blocking=True)
model.load_state_dict(torch.load("../out/models/e4/e4_13_acc_0.83_loss_0.55229"))
# train_t(model, dataset_planetoid_graph, EPOCHS, device, 1e-2, torch.nn.CrossEntropyLoss())
evaluate_t(model, dataset_planetoid_graph, device, torch.nn.CrossEntropyLoss(), dataset_planetoid.num_classes)
