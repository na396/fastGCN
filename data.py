
import warnings
import math
import numpy as np
import torch

from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected
from torch_geometric.utils import index_to_mask
from torch_geometric.utils import degree

from collections import Counter
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.sparse import spdiags

######################################################################################################################### def inside
def inside(feature, data):
    if hasattr(data, feature):
        cap = str(feature) + ": "
        print(cap, getattr(data, feature))
    else:
        print("data does not have ", feature)

######################################################################################################################### def createMask
def createMask(data, split_idx):

    # create train, val, test mask for ogb data
    # data: ogb graph
    # split_idx: its split

    train_mask, val_mask, test_mask = split_idx["train"], split_idx["valid"], split_idx["test"]
    mask_orig = torch.arange(start=0, end=data.x.shape[0])
    data.train_mask = torch.isin(mask_orig, train_mask)
    data.val_mask = torch.isin(mask_orig, val_mask)
    data.test_mask = torch.isin(mask_orig, test_mask)

    return data

######################################################################################################################### def data_preparation
def data_prepare(dataset_name, data_dir, maskInd=None):
    # download, and prepare the semi-supervised graphs for node classification

    ans = {}
    if dataset_name == "Cora":
        from torch_geometric.datasets import Planetoid

        cora = Planetoid(name='Cora',
                         root=data_dir)
        print(str("Cora dataset of ") + str(len(cora)) + str(" graphs"))
        cora = cora[0]
        cora.graph_name = "Cora"
        cora = data_summary(cora, "Cora")
        cora = data_cleaning(cora, maskInd)
        print("\n")
        cora = data_summary(cora, "Cora")
        ans = cora
        print("\n")

    if dataset_name == "CiteSeer":
        from torch_geometric.datasets import Planetoid

        citeseer = Planetoid(name='CiteSeer',
                             root=data_dir)
        print(str("CiteSeer dataset of ") + str(len(citeseer)) + str(" graphs"))
        citeseer = citeseer[0]
        citeseer.graph_name = "CiteSeer"
        citeseer = data_summary(citeseer, "CiteSeer")
        print("\n")
        citeseer = data_cleaning(citeseer, maskInd)
        citeseer = data_summary(citeseer, "CiteSeer")
        ans = citeseer
        print("\n")

    if dataset_name == "PubMed":
        from torch_geometric.datasets import Planetoid

        pubmed = Planetoid(name='PubMed',
                           root=data_dir, )
        print(str("PubMed dataset of ") + str(len(pubmed)) + str(" graphs"))
        pubmed = pubmed[0]
        pubmed.graph_name = "PubMed"
        pubmed = data_summary(pubmed, "PubMed")
        print("\n")
        pubmed = data_cleaning(pubmed, maskInd)
        pubmed = data_summary(pubmed, "PubMed")
        ans = pubmed
        print("\n")

    if dataset_name == "WikiCs":
        from torch_geometric.datasets import WikiCS

        wikics = WikiCS(root=data_dir)
        print(str("WikiCs dataset of ") + str(len(wikics)) + str(" graphs"))
        wikics = wikics[0]
        wikics.graph_name = "WikiCs"
        wikics = data_summary(wikics, "WikiCs")
        print("\n")
        wikics = data_cleaning(wikics, maskInd)
        wikics = data_summary(wikics, "WikiCs")
        ans = wikics
        print("\n")

    if dataset_name == "Arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset

        arxive = PygNodePropPredDataset(name="ogbn-arxiv",
                                        root=data_dir)
        arxive = createMask(arxive[0], arxive.get_idx_split())
        print(str("Arxiv dataset of ") + str(len(arxive)) + str(" graphs"))
        arxive.graph_name = "Arxiv"
        arxive = data_summary(arxive, "Arxiv")
        print("\n")
        arxive = data_cleaning(arxive, maskInd)
        arxive = data_summary(arxive, "Arxiv")
        ans = arxive
        print("\n")

    if dataset_name == "Products":
        from ogb.nodeproppred import PygNodePropPredDataset

        products = PygNodePropPredDataset(name="ogbn-products",
                                        root=data_dir)
        print(str("Products dataset of ") + str(len(products)) + str(" graphs"))
        products = createMask(products[0], products.get_idx_split())
        products.graph_name = "Products"
        products = data_summary(products, "Products")
        print("\n")
        products = data_cleaning(products, maskInd)
        products = data_summary(products, "Products")
        ans = products
        print("\n")

    return ans

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
            if data.train_mask.dim() > 1:
                print("number of different train split is ", data.train_mask.size(1))
            else:
                print("number of different train split is ", 1)
            if data.train_mask.dim() == 1:
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

######################################################################################################################### def drpSmallCCnodes
def data_cleaning(data, maskInd=None):

    print("graph ", data.graph_name)
    x, edge_index, edge_attr, num_node = data.x, data.edge_index, data.edge_attr, data.num_nodes

    if data.has_self_loops():
        print("removing self loops...")
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    if data.is_directed():
        print("making graph " + str(data.graph_name) + " undirected...")
        edge_index = to_undirected(edge_index)

    A = to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=num_node).tocsr()
    cc = connected_components(A, directed=False)[1]
    cc = np.array(cc)
    freq = Counter(cc)
    cc_label = max(freq, key=freq.get)
    nodes = np.where(cc == cc_label)[0]
    A = A[:, nodes][nodes, :]

    edge_index, _ = from_scipy_sparse_matrix(A)
    nodes = torch.tensor(nodes)
    mask = index_to_mask(nodes)
    data.edge_attr = edge_attr
    data.edge_index = edge_index
    data.x = x[nodes]
    data.y = data.y[nodes]

    if maskInd==None: maskInd = torch.randint(0, data.train_mask.shape[1], (1,))

    if data.train_mask.dim() == 1:
        data.train_mask = data.train_mask[nodes]
    else:
        data.train_multipleMask = data.train_mask[nodes, :]
        data.train_mask = data.train_mask[nodes, maskInd]

    if data.val_mask.dim() == 1:
        data.val_mask = data.val_mask[nodes]
    else:
        data.val_multipleMask = data.val_mask[nodes, :]
        data.val_mask = data.val_mask[nodes, maskInd]

    if data.test_mask.dim() == 1:
        data.test_mask = data.test_mask[nodes]
    else:
        data.test_multipleMask = data.test_mask[nodes, :]
        data.test_mask = data.test_mask[nodes, maskInd]

    data.num_nodes = data.x.shape[0]
    data.num_edges = data.edge_index.shape[1]
    data.classes = torch.unique(data.y)
    data.num_classes = len(data.classes)
    data.node_mask = mask

    return data

######################################################################################################################### def spectral_embedding
def spectral_embedding(data, dataset_name, ncol=0, drp_first=True, use_cache = False):

    # calculates spectral embedding
    # data: graph
    # dataset_name: name of the dataset, either "Cora", "CiteSeer", "PubMed", "WikiCs", "Arxiv", "Products"

    k = data.num_nodes if ncol == 0 else ncol+1

    x, edge_index = data.x, data.edge_index
    deg = degree(edge_index[0])
    A = to_scipy_sparse_matrix(edge_index=edge_index).tocsr()
    D = spdiags(1/deg.sqrt(), 0, data.num_nodes, data.num_nodes)

    DA = D.dot(A)
    L = DA.dot(D)

    if use_cache:
        X = torch.load(f'Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/Cache/eigval_embedding_{dataset_name}.pt', map_location=torch.device('cpu'))
        Y = torch.load(f'Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/Cache/eigvec_embedding_{dataset_name}.pt', map_location=torch.device('cpu'))
    else:
        X, Y = eigsh(A=L, k=k, which='LM')
        torch.save(X, f'Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/Cache/eigval_embedding_{dataset_name}.pt')
        torch.save(Y, f'Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/Cache/eigvec_embedding_{dataset_name}.pt')

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    Xs = X.sort(descending=True)
    X = Xs.values
    Y = Y[:, Xs.indices]

    # D = torch.diag(1. / torch.sqrt(deg))
    D  = torch.sparse.spdiags(1/deg.sqrt(), torch.tensor([0]), (data.num_nodes, data.num_nodes))
    Y = torch.matmul(D, Y) #row_wise multiplication each row of Y with 1/sqrt(deg)
    Y = torch.sub(Y, torch.matmul(deg, Y) / deg.sum())

    # D = torch.diag(deg)
    D = torch.sparse.spdiags(deg, torch.tensor([0]), (data.num_nodes, data.num_nodes))
    for j in range(Y.size(1)):
        x = Y[:, j]
        Y[:,j] = x/torch.sqrt(torch.matmul(torch.mul(x, deg), x))

    if drp_first: Y = Y[:, 1:Y.size(1)]
    constant = torch.matmul(torch.mul(Y[:, 0], deg), Y[:,0])
    if torch.round(constant, decimals=5) != 1:
        warnings.warn("constant condition does not hold! ")

    data.eigenvectors = Y
    data.eigenvalues = X[1:len(X)]

    if (torch.round(data.eigenvalues, decimals=5)==1).sum() >1:
        raise ValueError('eigenvalues equal to 1 is more than 1 in spectral_embedding...')

    return data

######################################################################################################################### def trainValidation_inClassSplit
def trainValidationTest_inClassSplit(data, label, trainVal_percent, train_percent, train_num, val_num):

    # data: graph, an object of dataset in torch_geometric
    # label: the label of the class
    # trainVal_percent: percentage for train and val set together, n - trainVal_percent: is the percent for test set
    # train_percent: percentage for train set
    # train_num: number of points for train set
    # val_percent: percentage for validation set out of training set
    # val_num: number of points for validation set
    # the remaining will be used for test


    y = (data.y == label).nonzero(as_tuple=True)[0]
    n = len(y)
    if trainVal_percent != None:
        train_val_num = math.ceil(n * trainVal_percent)
        train_num = math.ceil(train_val_num * train_percent)
        val_num = train_val_num - train_num


    suffle_inds = torch.randperm(len(y))
    y = y[suffle_inds]

    perm_train = y[0:train_num]
    perm_val = y[train_num:train_num + val_num]
    perm_test = y[train_num + val_num:]

    return perm_train, perm_val, perm_test

######################################################################################################################### def trainValidationTest_splitPerClass
def trainValidationTest_splitPerClass(data, trainVal_percent, train_percent, train_num=None, val_num=None,
                                      verbose=True):

    # data: graph
    # train_percent: percentage for train set
    # train_num: number of points for train set
    # val_percent: percentage for validation set
    # val_num: number of points for validation set
    # the remaining will be used for test

    ################################################################

    ################################################################
    train_all = val_all = test_all = torch.tensor([])

    for lab in torch.unique(data.y):
        perm_train, perm_val, perm_test = trainValidationTest_inClassSplit(data, lab,
                                                                           trainVal_percent, train_percent,
                                                                           train_num, val_num)
        train_all = torch.cat((train_all, perm_train))
        val_all = torch.cat((val_all, perm_val))
        test_all = torch.cat((test_all, perm_test))

    indices = torch.arange(0, data.num_nodes)
    trainMask_perClass = torch.isin(indices, train_all)
    valMask_perClass = torch.isin(indices, val_all)
    testMask_perClass = torch.isin(indices, test_all)

    data.trainMask_perClass = trainMask_perClass
    data.valMask_perClass = valMask_perClass
    data.testMask_perClass = testMask_perClass

    if verbose:
        print(data.graph_name)
        print("trainMask_perClass size: ", trainMask_perClass.size())
        print("trainMask_perClass.sum().item(): ", trainMask_perClass.sum().item())
        print("valMask_perClass size: ", valMask_perClass.size())
        print("valMask_perClass.sum().item(): ", valMask_perClass.sum().item())
        print("testMask_perClass size: ", testMask_perClass.size())
        print("testMask_perClass.sum().item(): ", testMask_perClass.sum().item())
        print("\n")

    return data

######################################################################################################################### def trainTest_split
def trainValidationTest_splitAllClasses(data, train_percent=None, train_num=None, val_percent=None, val_num=None,
                                        verbose=True):

    # data: graph
    # train_percent: percentage for train set
    # train_num: number of points for train set
    # val_pecent: percentage for validation set
    # val_num: number of points for validation set
    # the remaining will be used for test

    ################################################################
    if (train_percent is not None) and (train_num is not None):
        warnings.warn('both train_percent and train_num are not None \n, making train_num to None...')
        train_num = None

    if (train_percent is None) and (train_num is None):
        raise TypeError("both percent and num cannot be None")

    if (val_percent is not None) and (val_num is not None):
        warnings.warn('both val_percent and val_num are not None \n, making val_num to None...')
        val_num = None

    if (val_percent is None) and (val_num is None):
        raise TypeError("both percent and num cannot be None")

    ################################################################
    if train_percent is not None:
        train_num = int(data.num_nodes * train_percent)


    if val_percent is not None:
        val_num = int(data.num_nodes * val_percent)

    ################################################################
    perm = torch.randperm(data.num_nodes)
    perm_train = perm[0:train_num]
    perm_val = perm[train_num:train_num+val_num]
    perm_test = perm[train_num+val_num:]

    indices = torch.arange(0, data.num_nodes)
    trainMask_allClasses = torch.isin(indices, perm_train)
    valMask_allClasses = torch.isin(indices, perm_val)
    testMask_allClasses = torch.isin(indices, perm_test)

    data.trainMask_allClasses = trainMask_allClasses
    data.valMask_allClasses = valMask_allClasses
    data.testMask_allClasses = testMask_allClasses

    if verbose:
        print(data.graph_name)
        print("trainMask_allClasses size: ", trainMask_allClasses.size())
        print("trainMask_allClasses.sum().item(): ", trainMask_allClasses.sum().item())
        print("valMask_allClasses size: ", valMask_allClasses.size())
        print("valMask_allClasses.sum().item(): ", valMask_allClasses.sum().item())
        print("testMask_allClasses size: ", testMask_allClasses.size())
        print("testMask_allClasses.sum().item(): ", testMask_allClasses.sum().item())
        print("\n")

    return data


