import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils.convert import to_networkx


def visualize(dataset):
    for i in range(10, 16):
        mnist = to_networkx(dataset[i])

    dataset.y = torch.zeros(75)
    node_labels = dataset.y.numpy()

    plt.figure(1, figsize=(5, 5))
    nx.draw(mnist, cmap=plt.get_cmap('Set2'), node_color=node_labels, node_size=30, linewidths=6)
    plt.show()


def visualize_t(dataset):
    graph = to_networkx(dataset[0])

    node_labels = dataset[0].y[list(graph.nodes)].numpy()

    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(14, 12))
    nx.draw(graph, cmap=plt.get_cmap('Blues'), node_color=node_labels, node_size=25, linewidths=3)
    plt.show()

