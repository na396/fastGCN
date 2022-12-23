
import warnings

import numpy as np
import networkx as nx
from networkx import from_numpy_matrix

import torch
from torch.linalg import eigh

from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected
from torch_geometric.utils import index_to_mask
from torch_geometric.utils import degree

######################################################################################################################### def inside
def inside(feature, data):
    if hasattr(data, feature):
        cap = str(feature) + ": "
        print(cap, getattr(data, feature))
    else:
        print("data does not have ", feature)

######################################################################################################################### def data_summary
def data_summary(data, gname, verbose = True):

    # prints the summary of the Data object in PyTorch
    # num_nodes: number of nodes
    # num_node_features: number of node features
    # num_edges: number of edges
    # edge_attr: number of edge features
    # num_classes: number of classes for manily Graph Classification
    # has_isolated_nodes: if True, graph has isolated node
    # has_self_loops: if True, nodes in graph has self loops
    # is_directed/undirected: if True, graph is directed/undirected
    # is_coalesced: if True, graph is coalesced (subgraphe)

    if not hasattr(data, "num_nodes"): data.num_nodes = data.x.shape[0]
    if not hasattr(data, "num_edges"): data.num_edges = data.edge_index.shape[1]
    if not hasattr(data, "num_classes"):
        data.classes = torch.unique(data.y)
        data.num_classes = len(data.classes)
    if data.y.dim() >1:
        data.y = torch.squeeze(data.y)

    if verbose:
        print("summary of " + str(gname))
        print(data)
        inside("num_nodes", data)
        inside("num_node_features", data)
        inside("num_edges", data)
        inside("edge_attr", data)
        inside("num_classes", data)
        print("has_isolated_nodes ", data.has_isolated_nodes())
        print("has_self_loops ", data.has_self_loops())
        print("is_directed ", data.is_directed())
        print("is_undirected ", data.is_undirected())
        print("is_coalesced ", data.is_coalesced())

        if hasattr(data, "train_mask"):
            if data.train_mask.dim() >1:
                print("number of different train split is ", data.train_mask.size(1))
            else:
                print("number of different train split is ", 1)
            if data.train_mask.dim() ==1:
                print("train_mask.sum().item() ", data.train_mask.sum().item())
            else:
                print("train_mask.sum().item() for first column", data.train_mask[:, 0].sum().item())
            if data.val_mask.dim() == 1:
                print("val_mask.sum().item() ", data.val_mask.sum().item())
            else:
                print("val_mask.sum().item() for first column", data.val_mask[:, 0].sum().item())
            if data.test_mask.dim() == 1:
                print("test_mask.sum().item() ", data.test_mask.sum().item())
            else:
                print("test_mask.sum().item() for first column", data.test_mask[:, 0].sum().item())

        print("num_classes", data.num_classes)
        print("classes", data.classes)


    return data

######################################################################################################################### def data_preparation
def data_prepare(dataset_name):
    # download, and prepare the semi-supervised graphs for node classification

    ans = {}
    if "Cora" in dataset_name:
        from torch_geometric.datasets import Planetoid

        cora = Planetoid(name='Cora',
                         root='/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets')
        print(str("Cora dataset of ") + str(len(cora)) + str(" graphs"))
        cora = cora[0]
        cora.graph_name = "Cora"
        cora = data_summary(cora, "Cora")
        cora = data_cleaning(cora)
        print("\n")
        cora = data_summary(cora, "Cora")
        ans["Cora"] = cora
        print("\n")

    if "CiteSeer" in dataset_name:
        from torch_geometric.datasets import Planetoid

        citeseer = Planetoid(name='CiteSeer',
                             root='/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets')
        print(str("CiteSeer dataset of ") + str(len(citeseer)) + str(" graphs"))
        citeseer = citeseer[0]
        citeseer.graph_name = "CiteSeer"
        citeseer = data_summary(citeseer, "CiteSeer")
        print("\n")
        citeseer = data_cleaning(citeseer)
        citeseer = data_summary(citeseer, "CiteSeer")
        ans["CiteSeer"] = citeseer
        print("\n")

    if "PubMed" in dataset_name:
        from torch_geometric.datasets import Planetoid

        pubmed = Planetoid(name='PubMed',
                           root='/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets', )
        print(str("PubMed dataset of ") + str(len(pubmed)) + str(" graphs"))
        pubmed = pubmed[0]
        pubmed.graph_name = "PubMed"
        pubmed = data_summary(pubmed, "PubMed")
        print("\n")
        pubmed = data_cleaning(pubmed)
        pubmed = data_summary(pubmed, "PubMed")
        ans["Pubmed"] = pubmed
        print("\n")

    if "WikiCs" in dataset_name:
        from torch_geometric.datasets import WikiCS

        wikics = WikiCS(root='/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets')
        print(str("WikiCs dataset of ") + str(len(wikics)) + str(" graphs"))
        wikics = wikics[0]
        wikics.graph_name = "WikiCs"
        wikics = data_summary(wikics, "WikiCs")
        print("\n")
        wikics = data_cleaning(wikics)
        wikics = data_summary(wikics, "WikiCs")
        ans["WikiCs"] = wikics
        print("\n")

    if "Arxive" in dataset_name:
        from ogb.nodeproppred import PygNodePropPredDataset

        arxive = PygNodePropPredDataset(name="ogbn-arxiv",
                                        root='/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets')
        print(str("Arxive dataset of ") + str(len(arxive)) + str(" graphs"))
        arxive = arxive[0]
        arxive.graph_name = "Arxive"
        arxive = data_summary(arxive, "Arxive")
        print("\n")
        arxive = data_cleaning(arxive)
        arxive = data_summary(arxive, "Arxive")
        ans["Arxive"] = arxive
        print("\n")

    if "Products" in dataset_name:
        from ogb.nodeproppred import PygNodePropPredDataset

        products = PygNodePropPredDataset(name="ogbn-products",
                                        root='/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets')
        print(str("Products dataset of ") + str(len(products)) + str(" graphs"))
        products = products[0]
        products.graph_name = "Products"
        products = data_summary(products, "Products")
        print("\n")
        products = data_cleaning(products)
        products = data_summary(products, "Products")
        ans["Products"] = products
        print("\n")

    return ans

######################################################################################################################### def zeroInds
def to_np_adjacencyMatrix(edge_index):

    adja = to_dense_adj(edge_index)
    adja = torch.squeeze(adja)
    adja = adja.numpy()

    return adja

######################################################################################################################### def largestCC
def largestCC(edge_index) -> torch.tensor:

    # net: an object of networkx
    # returns the index of nodes in the largest connceted components for an undirected graph

    A = to_np_adjacencyMatrix(edge_index)
    net = from_numpy_matrix(A)
    cc = [net.subgraph(c).copy() for c in nx.connected_components(net)]
    cc_len = [len(c.nodes) for c in cc]
    lcc = int(np.argmax(cc_len))
    keep_NodeInds = list(cc[lcc].nodes)

    return torch.tensor(keep_NodeInds).int().long()

######################################################################################################################### def drpSmallCCnodes
def data_cleaning(data):

    print("graph ", data.graph_name)
    x, edge_index, edge_attr, num_node = data.x, data.edge_index, data.edge_attr, data.num_nodes

    if data.has_self_loops():
        print("removing self loops...")
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # if data.has_isolated_nodes():
    #     print("removing isolated nodes...")
    #     num_nodes, edge_index, mask_nodes = remove_isolated_nodes(edge_index)

    if data.is_directed():
        print("making graph " + str(data.graph_name) + " undirected...")
        edge_index = to_undirected(edge_index)

    nodes = largestCC(edge_index)
    mask = index_to_mask(nodes)

    A = to_np_adjacencyMatrix(edge_index)
    A = A[nodes.numpy()[:, None], nodes.numpy()]
    A = torch.tensor(A)
    A = torch.unsqueeze(A, dim=0)

    data.edge_index, _ = dense_to_sparse(A)
    data.edge_attr = edge_attr
    data.x = x[nodes]
    data.y = data.y[nodes]

    data.train_mask = data.train_mask[nodes] if data.train_mask.dim()==1 else  data.train_mask[nodes,: ]
    data.val_mask = data.val_mask[nodes] if data.val_mask.dim() == 1 else data.val_mask[nodes,:]
    data.test_mask = data.test_mask[nodes] if data.test_mask.dim() == 1 else data.test_mask[nodes, :]

    data.num_nodes = data.x.shape[0]
    data.num_edges = data.edge_index.shape[1]
    data.classes = torch.unique(data.y)
    data.num_classes = len(data.classes)
    data.node_mask = mask

    return data

######################################################################################################################### def spectral_embedding
def spectral_embedding(data, drp_first=True):

    # data: graph

    x, edge_index = data.x, data.edge_index
    A = to_dense_adj(edge_index)
    A = torch.squeeze(A)

    deg = degree(edge_index[0])
    D = torch.diag(1. /torch.sqrt(deg))

    DA = torch.matmul(D,A)
    L = torch.matmul(DA, D)
    eigs = eigh(L)
    X = eigs.eigenvalues
    Y = eigs.eigenvectors

    Xs = X.sort(descending=True)
    X = Xs.values
    Y = Y[:, Xs.indices]

    Y = torch.matmul(D, Y)
    Y = torch.sub(Y, torch.matmul(deg, Y)/deg.sum())

    D = torch.diag(deg)

    for j in range(Y.size(1)):
        x = Y[:, j]
        Y[:,j] = x/torch.sqrt(torch.matmul(torch.matmul(x, D), x))

    if drp_first: Y = Y[:, 1:Y.size(1)]
    constant = torch.matmul(torch.matmul(Y[:, 0], D), Y[:,0])
    if torch.round(constant, decimals=5) != 1:
        warnings.warn("constant condition does not hold! ")

    data.eigenvectors = Y
    data.eigenvalues = X

    if (torch.round(data.eigenvalues, decimals=5)==1).sum() >1:
        raise ValueError('eigenvalues equal to 1 is more than 1 in spectral_embedding...')

    return data



