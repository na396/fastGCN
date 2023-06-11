
import gc
from spectrumMLP import *
from data import *
from train import *
from utils import *
from parameters_spectrumMLP import *

ncol=100
root_dir = "Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code"

dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs"] # dataset name
#dataset_name = ["Cora"]

for ds in dataset_name:
    gc.collect()

    graph = data_prepare(dataset_name=ds, maskInd=maskInd, root_dir=root_dir)

    graph = trainValidationTest_splitPerClass(data=graph, trainVal_percent=trainVal_percent_perClass,
                                              train_percent=train_percent_perClass,
                                              train_num=train_num_perClass, val_num=val_num_perClass,
                                              verbose=data_verobse)

    graph = trainValidationTest_splitAllClasses(graph, train_percent_allClasses, train_num_allClasses,
                                                val_percent_allClasses, val_num_allClasses,
                                                verbose=data_verobse)

    k = graph.num_nodes if ncol == 0 else ncol + 1

    x, edge_index = graph.x, graph.edge_index
    deg = degree(edge_index[0])
    A = to_scipy_sparse_matrix(edge_index=edge_index).tocsr()
    D = spdiags(1 / deg.sqrt(), 0, graph.num_nodes, graph.num_nodes)

    DA = D.dot(A)
    L = DA.dot(D)
    X, Y = eigsh(A=L, k=k, which='LM')

    cache_dir_egval = root_dir + "/Cache/eigval_embedding_" + "Cora" + ".pt"
    cache_dir_egvec = root_dir + "/Cache/eigvec_embedding_" + "Cora" + ".pt"

    torch.save(X, cache_dir_egval)
    torch.save(Y, cache_dir_egvec)