"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""

globals().clear()
######################################################################################################################## libraries
import gc
import json
from spectrumMLP import *
from train import *
from data import *
from utils import *
from parameters_spectrumMLP import *

######################################################################################################################### hyper-parameter initialization
#root_dir = "Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code"
#root_dir = "/home/n/na396/fastGCN" # directory to store the data
#root_dir = '/content/drive/MyDrive/Code'
#root_dir = "D:/Niloo's Project"

#root_dir = "C:/Users/ikoutis/Documents/GitHub/fastGCN"
root_dir = "/home/ryanlee/Documents/fastGCN"
#########################################################################################################################

#dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxiv", "Products"] # dataset name
dataset_name = ["Cora"]

######################################################################################################################### results df
epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF() # summary of trained model for train, validation, and test, both accuracy and loss
graphs = {}

torch.cuda.empty_cache()
gc.collect()

device = gpu_setup(use_gpu)

if device.type == 'cuda':
  torch.cuda.manual_seed(123)
  torch.cuda.manual_seed_all(123)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


######################################################################################################################### data loading and preperation
# graph: a list of prepared graph datasets

learning_rate = 0.01
num_epoch = 2000
num_exp = 10                                       # number of experiments
use_cache = True
embedding_list = ["non-symmetric", "symmetric"]    # "deepwalk can be added"
coeff_list = [2,4]
weight_decay_list = [0]                            # add other real number is needed
highest_validation = {}                            #Multi-nested Dictionary: first index denotes coeff, second index denotes embedding, third denotes experiment number, fourth denotes every "delta" epochs which records the highest valid accuracy with corresponding test accuracy
                                                   

script_number = 10
epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF()  # summary of trained model for train, validation, and test, both accuracy and loss

for coeff in coeff_list:
    highest_validation[coeff] = {}
    for embed in embedding_list:
        highest_validation[coeff][embed] = {}
        script_number += 1
        name = f"coeff={coeff}+embed={embed}"
        print(f"{name} \n")
        check_directory(f"{root_dir}/Results/Script{script_number}/")
        f = open(f"{root_dir}/Results/Script{script_number}/model_name.txt", "w")
        f.write(name)
        f.close()

        ##################################################### datasets
        for ds in dataset_name:
            gc.collect()

            graph = data_prepare(dataset_name=ds, maskInd=maskInd, root_dir=root_dir)

            graph = trainValidationTest_splitPerClass(data=graph, trainVal_percent=trainVal_percent_perClass,
                                                          train_percent=train_percent_perClass,
                                                          train_num=train_num_perClass, val_num=val_num_perClass,
                                                          verbose=data_verbose)

            graph = trainValidationTest_splitAllClasses(graph, train_percent_allClasses, train_num_allClasses,
                                                            val_percent_allClasses, val_num_allClasses,
                                                            verbose=data_verbose)

            graph.dir = root_dir

            if embed in ["non-symmetric", "symmetric"]:
                graph = embedding(data=graph, dataset_name=graph.graph_name, root_dir=graph.dir,
                                    ncol=coeff * graph.num_nodes, drp_first=True, use_cache=use_cache, mode=embed)

            elif embed=="deepwalk":
                graph.embedding_vectors = deepwalk(data=graph, root_dir=root_dir, dataset_name=ds,
                                                       emb_dim=deepwalk_emb_dim, learning_rate=deepwalk_lr,
                                                       n_epoch=deepwalk_epoch, mask_type="original",
                                                       batch_size=deepwalk_batchSize)

            for iter_num in range(num_exp):

                ##################################################### model: spectrumMLP
                gt = time.time()
                torchStatus()
                torch.cuda.seed_all()
                print("\n")

                ##################################################### initialization

                graph = graph.to(device)

                st = time.time()
                mdl = spectrumMLP(in_dim=coeff*graph.num_classes, hidden_dim=4*graph.num_classes,
                                    out_dim=graph.num_classes, num_lin=num_linear, add_relu=add_relu,
                                    bias=mlp_bias, init=init, pdrop=pdrop, embedding=embed,
                                    deepwalk_epoch=deepwalk_epoch, deepwalk_lr=deepwalk_lr,
                                    deepwalk_maskType=deepwalk_maskType, deepwalk_batchSize=deepwalk_batchSize,
                                    cache=use_cache).to(device)

                print(f"model is {mdl.model_name} \n")
                mdl_name = mdl.model_name
                opt = torch.optim.Adam(mdl.parameters(), lr=learning_rate)

                ##################################################### training phase
                print("entering training phase...\n")
                mdl, opt, epochDF, deltaResults = train(model=mdl, optimizer=opt, mask_type=mask_type,
                                                        num_epoch=num_epoch, data=graph, keepResult=train_keep,
                                                        verbose=train_verbose, iter_num=iter_num)
                highest_validation[coeff][embed][iter_num+1]=deltaResults
                t_train = time.time() - st
                plot_epoch(df=epochDF, graph_name=graph.graph_name, root_dir=root_dir, sc_num=script_number,
                            col="loss", keep=plt_keep, sh=plt_sh, imagetype=imagetype, iter_num=iter_num)
                plot_epoch(df=epochDF, graph_name=graph.graph_name, root_dir=root_dir, sc_num=script_number,
                            col="accuracy", keep=plt_keep, sh=plt_sh, imagetype=imagetype, iter_num=iter_num)

                epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

                print("\n")

                ########################################## end of training phase

                ##################################################### test phase
                print("entering test phase...\n")
                st = time.time()

                mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type=mask_type,
                                    keepResult=test_keep, verbose=test_verbose, iter_num=iter_num)
                t_test = time.time() - st
                summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

                ft = time.time() - gt
                ##################################################### saving results
                print("saving results...\n")
                ans = {"name": name, 'model_name': mdl_name, "model": mdl, "optimizer": opt,
                        'epochResults': epochDF, 'summaryResults': sumDF,
                        "t_train": t_train, "t_test": t_test, "t_all": gt,
                        'iteration_number':iter_num}

                ##########################################################################################  save the result for each graph
                check_directory(
                    f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{iter_num}/")

                torch.save(ans,
                            f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{iter_num}/{graph.graph_name}_{iter_num}.pt")

                pd_dir = f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{iter_num}/{graph.graph_name}_{iter_num}_epochResults.pkl"
                epochDF.to_pickle(pd_dir)

                pd_dir = f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{iter_num}/{graph.graph_name}_{iter_num}_summaryResults.pkl"
                sumDF.to_pickle(pd_dir)

                del mdl

            del graph
            torch.cuda.empty_cache()
            gc.collect()

                ##########################################################################################  save the result for all graphs

with open(f"{root_dir}/Results/highest_Validation.txt", "w") as fp:
    json.dump(highest_validation, fp)
epochResults.to_pickle(f"{root_dir}/Results/all_experiment_spectrumMLP_epochResults.pkl")
summaryResults.to_pickle(f"{root_dir}/Results/all_experiment_spectrumMLP_summaryResults.pkl")