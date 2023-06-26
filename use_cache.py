
import gc
from spectrumMLP import *
from data import *
from train import *
from utils import *
from parameters_spectrumMLP import *

learning_rate = 0.01
num_epoch = 5
use_cache = True

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

    embed = deepwalk(data=graph, emb_dim=deepwalk_emb_dim, learning_rate=deepwalk_lr,
                        n_epoch=deepwalk_epoch, mask_type="original", batch_size=deepwalk_batchSize)

    cache_dir = root_dir + "/Cache/deepwalk_embedding_" + str(dataset_name) + ".pt"
    torch.save(embed, cache_dir)
