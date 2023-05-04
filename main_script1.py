r"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""

globals().clear()
######################################################################################################################## libraries
import pickle
import gc

from data import *
from spectrumGCN import *
from train import *
from utils import *

from paramteres import *
######################################################################################################################### hyper-parameter initialization
torch.manual_seed(123)

p_dropout = 0.5
graph_less = True
multiple_concat = True
eg_features = True

learning_rate = 0.001
num_epoch = 500
use_cache = False
script_number = "1"


root_dir = "Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/"
#root_dir = "/home/n/na396/fastGCN/" # directory to store the data
#root_dir = '/content/drive/MyDrive/Code/'
#root_dir = "D:/Niloo's Project/"

#########################################################################################################################

#dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxiv", "Products"] # dataset name
dataset_name = ["Cora"]

######################################################################################################################### results df
epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF() # summary of trained model for train, validation, and test, both accuracy and loss
modelResults = {} # final model
optimizerResults = {} # final optimizer
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
    # temp = "WithOutGraph" if graph_less else 'WithGraph'
    # mdl_name = "spectrumGCN" + '+' + str(graph.graph_name) + '+' + str(temp)

    print(f"graph is {graph.graph_name}")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = graph.to(device)

    st = time.time()
    mdl = spectrumGCN(graph_less=graph_less,
                      spec_in=2 * graph.num_classes, spec_out=graph.num_classes,
                      conv1_in_dim=graph.num_node_features, conv1_out_dim=conv1_out,
                      conv2_out_dim=graph.num_classes,
                      num_linear=2, add_relu=True,
                      conv_bias=True, multiple_concatenations=multiple_concat,
                      pdrop=p_dropout, eg_features=eg_features, eg_features_dim=2*graph.num_classes).to(device)

    print(f"model is {mdl.model_name}")
    mdl_name = mdl.model_name
    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate)

    print("\n")
    ##################################################### training phase
    print("entering training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type=mask_type,
                              num_epoch=num_epoch, data=graph, keepResult=train_keep, verbose=train_verbose)
    t_train = time.time() - st
    plot_epoch(df=epochDF, model_name=mdl_name, root_dir=root_dir, sc_num=script_number,
               col="loss", keep=plt_keep, sh=plt_sh, imagetype=imagetype)
    plot_epoch(df=epochDF, model_name=mdl_name, root_dir=root_dir, sc_num=script_number,
               col="accuracy", keep=plt_keep, sh=plt_sh, imagetype=imagetype)

    epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ########################################## end of training phase

    ##################################################### test phase
    print("entring test phase...\n")
    st = time.time()

    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type=mask_type, keepResult=test_keep,
                      verbose=test_verbose)
    t_test = time.time() - st
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ft = time.time() - gt
    ##################################################### saving results
    print("saving results...")
    res = {'model_name': mdl_name, "model": mdl, "optimizer": opt,
           'epochResults': epochResults, 'summaryResults': summaryResults,
           "t_train": t_train, "t_test": t_test, "t_all": gt}

    ##########################################################################################  save the result
    pd_dir = root_dir + "Results/" + str(graph.graph_name) + str(script_number) + "_epochResults" + ".pkl"
    epochResults.to_pickle(pd_dir)

    pd_dir = root_dir + "Results/" + str(graph.graph_name) + str(script_number) + "_summaryResults" + ".pkl"
    summaryResults.to_pickle(pd_dir)

    dic_dir = root_dir + "Results/dic.pt"
    try:
        final = torch.load(dic_dir, map_location=torch.device('cpu'))
    except (OSError, IOError, EOFError) as e:
        final = {}

    final[str(graph.graph_name) + str(script_number)] = res
    torch.save(final, dic_dir)

    graph_dir = root_dir + "Results/graphs.pt"
    try:
        graphs = torch.load(graph_dir, map_location=torch.device('cpu'))
    except (OSError, IOError, EOFError) as e:
        graphs = {}

    graphs[str(graph.graph_name) + str(script_number)] = graph
    torch.save(graphs, graph_dir)

    del graph
    torch.cuda.empty_cache()
    gc.collect()

    print("\n\n\n\n")


