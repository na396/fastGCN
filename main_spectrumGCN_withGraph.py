r"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""

globals().clear()
######################################################################################################################## libraries
import gc

from data import *
from spectrumGCN import *
from train import *
from utils import *

from parameters_spectrumGCN import *
######################################################################################################################### hyper-parameter initialization
torch.manual_seed(123)

root_dir = "Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code"
#root_dir = "/home/n/na396/fastGCN" # directory to store the data
#root_dir = '/content/drive/MyDrive/Code'
#root_dir = "D:/Niloo's Project"

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

graph_less = False
learning_rate = 0.001
num_epoch = 5
use_cache = True
iteration_number = None

script_number = 8

epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF()  # summary of trained model for train, validation, and test, both accuracy and loss


for drop in [0.5, 0]:

    script_number += 1
    p_dropout = drop

    name = f"graphless={graph_less}+drop={drop}"
    check_directory(f"{root_dir}/Results/Script{script_number}/")
    f = open(f"{root_dir}/Results/Script{script_number}/model_name.txt", "w")
    f.write(name)
    f.close()

    for ds in dataset_name:
        gc.collect()

        graph = data_prepare(dataset_name=ds, maskInd=maskInd, root_dir=root_dir)

        graph = spectral_embedding(data=graph, dataset_name=ds, root_dir=root_dir, ncol=2 * graph.num_classes,
                                   drp_first=drp_first, use_cache=use_cache)

        graph = trainValidationTest_splitPerClass(data=graph, trainVal_percent=trainVal_percent_perClass,
                                                  train_percent=train_percent_perClass,
                                                  train_num=train_num_perClass, val_num=val_num_perClass,
                                                  verbose=data_verobse)

        graph = trainValidationTest_splitAllClasses(graph, train_percent_allClasses, train_num_allClasses,
                                                    val_percent_allClasses, val_num_allClasses,
                                                    verbose=data_verobse)

        ##################################################### model: spectrumGCN
        gt = time.time()
        torchStatus()
        torch.cuda.seed_all()
        print("\n")

        ##################################################### initialization
        print(f"graph is {graph.graph_name}")

        graph = graph.to(device)

        st = time.time()
        mdl = spectrumGCN(graph_less=graph_less,
                          spec_in=2 * graph.num_classes, spec_out=graph.num_classes, num_linear=num_linear,
                          add_relu=add_relu,
                          conv1_in_dim=graph.num_node_features, conv1_out_dim=conv1_out,
                          conv2_out_dim=graph.num_classes,
                          conv_bias=True, multiple_concatenations=False,
                          pdrop=p_dropout, eg_features=False).to(
            device)

        print(f"model is {mdl.model_name}")
        mdl_name = mdl.model_name
        opt = torch.optim.Adam(mdl.parameters(),
                               lr=learning_rate)

        print("\n")
        ##################################################### training phase
        print("entering training phase...\n")
        mdl, opt, epochDF = train(model=mdl, optimizer=opt, mask_type=mask_type,
                                  num_epoch=num_epoch, data=graph, keepResult=train_keep,
                                  verbose=train_verbose, iter_num=iteration_number)
        t_train = time.time() - st
        plot_epoch(df=epochDF, graph_name=graph.graph_name, root_dir=root_dir, sc_num=script_number,
                   col="loss", keep=plt_keep, sh=plt_sh, imagetype=imagetype)
        plot_epoch(df=epochDF, graph_name=graph.graph_name, root_dir=root_dir, sc_num=script_number,
                   col="accuracy", keep=plt_keep, sh=plt_sh, imagetype=imagetype)

        epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

        print("\n")

        ########################################## end of training phase

        ##################################################### test phase
        print("entering test phase...\n")
        st = time.time()

        mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type=mask_type, keepResult=test_keep,
                          verbose=test_verbose)
        t_test = time.time() - st
        summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

        ft = time.time() - gt
        ##################################################### saving results
        print("saving results...")
        ans = {"name": name, 'model_name': mdl_name, "model": mdl, "optimizer": opt,
               'epochResults': epochDF, 'summaryResults': sumDF,
               "t_train": t_train, "t_test": t_test, "t_all": gt}

        ##########################################################################################  save the result for each graph
        check_directory(
            f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/")

        torch.save(ans, f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{graph.graph_name}.pt")

        pd_dir = f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{graph.graph_name}_epochResults.pkl"
        epochDF.to_pickle(pd_dir)

        pd_dir = f"{root_dir}/Results/Script{script_number}/{graph.graph_name}/{graph.graph_name}_summaryResults.pkl"
        sumDF.to_pickle(pd_dir)

        del graph
        torch.cuda.empty_cache()
        gc.collect()

##########################################################################################  save the result for all graphs
epochResults.to_pickle(f"{root_dir}/Results/all_experiment_without_graph_epochResults.pkl")
summaryResults.to_pickle(f"{root_dir}/Results/all_experiment_without_graph_summaryResults.pkl")
