r"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""

######################################################################################################################## libraries

from data import *
from classGCN import *
from train import *
from utils import *


######################################################################################################################### hyper-parameter initialization
torch.manual_seed(123)

#
train_percent = 0.65
val_percent = 0.15
test_percent = 0.2
#
train_num = None
val_num = None
test_num = None
#
weight_decay =5e-4
learning_rate = 0.01
num_epoch = 5000
h1_dim = 128
h2_dim = 128

pdrop= 0.5 # dropout percentage

#spec_in_dim = graph.num_classes
#spec_out_dim = 2*graph.num_classes

#dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxive", "Products"]
dataset_name = ["Cora"]


######################################################################################################################### data loading and preperation
# graph: a list of prepared graph datasets
graphs = data_prepare(dataset_name)
graphs_name = list(graphs.keys())
for graph in graphs_name:
    graphs[graph] = spectral_embedding(graphs[graph], drp_first=True)
    graphs[graph] = trainValidationTest_split(graphs[graph], train_percent, train_num, val_percent, val_num, verbose=True)

######################################################################################################################### resuls df
epochResults = epochPerformanceDF()
summaryResults = TrainValidationTestDF()
modelResuls = {}
optimizerResults = {}
######################################################################################################################### model: vanilaGCN

mdl_name = "vanilaGCN"
for key in graphs_name:
    ##################################################### initialization
    graph = graphs[key]
    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = vanilaGCN(in_dim=graph.num_node_features,
                    h1_dim=h1_dim,
                    h2_dim=h2_dim,
                    out_dim=graph.num_classes,
                    pdrop=pdrop).to(device)

    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type="manualMask",
                                    num_epoch=num_epoch, data=graph, keepResult=True, verbose=False)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=True, sh=False)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=True, sh=False)

    results = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type="manualMask", keepResult=True,
                      verbose=True)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    modelResuls[mdl_name] = mdl
    optimizerResults[mdl_name] = opt

######################################################################################################################### model: spectrumGCN_inLayer

mdl_name = "spectrumGCN_inLayer"
for key in graphs_name:
    ##################################################### initialization
    graph = graphs[key]
    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = spectralGCN_inLayer(data=graph,
                      in_dim=graph.num_node_features,
                      h1_dim=h1_dim,
                      h2_dim=h2_dim,
                      out_dim=graph.num_classes,
                      spec_in_dim=graph.num_classes,
                      spec_out_dim=2*graph.num_classes,
                      lin=False, bias=False, allLayers=False, pdrop=pdrop).to(device)

    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type="manualMask",
                                    num_epoch=num_epoch, data=graph, keepResult=True, verbose=False)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=True, sh=False)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=True, sh=False)

    results = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type="manualMask", keepResult=True,
                      verbose=True)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    modelResuls[mdl_name] = mdl
    optimizerResults[mdl_name] = opt

######################################################################################################################### model: spectrumGCN_outLayer

mdl_name = "spectrumGCN_outLayer"
for key in graphs_name:
    ##################################################### initialization
    graph = graphs[key]
    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = spectralGCN_outLayer(data=graph,
                      in_dim=graph.num_node_features,
                      h1_dim=h1_dim,
                      h2_dim=h2_dim,
                      out_dim=graph.num_classes,
                      spec_in_dim=graph.num_classes,
                      spec_out_dim=2*graph.num_classes,
                      lin=True, bias=False, pdrop=pdrop).to(device)

    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type="manualMask",
                                    num_epoch=num_epoch, data=graph, keepResult=True, verbose=False)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=True, sh=False)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=True, sh=False)

    results = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type="manualMask", keepResult=True,
                      verbose=True)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    modelResuls[mdl_name] = mdl
    optimizerResults[mdl_name] = opt

######################################################################################################################### model: spectrumGCN_hiddenLayer

mdl_name = "spectrumGCN_hiddenLayer"
for key in graphs_name:
    ##################################################### initialization
    graph = graphs[key]
    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = spectralGCN_hiddenLayer(data=graph,
                      in_dim=graph.num_node_features,
                      h1_dim=h1_dim,
                      h2_dim=h2_dim,
                      out_dim=graph.num_classes,
                      spec_in_dim=graph.num_classes,
                      spec_out_dim=2*graph.num_classes,
                      lin=True, bias=False, allLayers=False, pdrop=pdrop).to(device)

    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type="manualMask",
                                    num_epoch=num_epoch, data=graph, keepResult=True, verbose=False)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=True, sh=False)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=True, sh=False)

    results = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type="manualMask", keepResult=True,
                      verbose=True)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    modelResuls[mdl_name] = mdl
    optimizerResults[mdl_name] = opt