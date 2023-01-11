r"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""

######################################################################################################################## libraries
import pickle
from data import *
from spectrumGCN import *
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
num_epoch = 10
#
mask_type="manualMask"
#
graph_less=False
# spec_in=2*graph.num_classes
# spec_out= graph.num_classes
num_linear = 2
add_relu = True
conv1_out = 128
conv_bias=True
p_dropout=0.5
#
train_verbose = True
train_keep = True

test_keep = True
test_verbose = True

plt_sh = False
plt_keep = True


#dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxive", "Products"]
dataset_name = ["Cora"]


######################################################################################################################### data loading and preperation
# graph: a list of prepared graph datasets
graphs = data_prepare(dataset_name)
graphs_name = list(graphs.keys())
for graph in graphs_name:
    graphs[graph] = spectral_embedding(graphs[graph], drp_first=True)
    graphs[graph] = trainValidationTest_split(graphs[graph], train_percent, train_num, val_percent, val_num, verbose=True)

######################################################################################################################### results df
epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF() # summary of trained model for train, validation, and test, both accuracy and loss
modelResults = {} # final model
optimizerResults = {} # final optimizer

######################################################################################################################### model: spectrumGCN

for key in graphs_name:

    ##################################################### initialization
    graph = graphs[key]
    temp = "GraphLess" if graph_less else 'WithGraph'
    mdl_name = "spectrumGCN" + '+' + str(graph.graph_name) + '+' + str(temp)

    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = spectrumGCN(graph_less=graph_less,
                 spec_in=2*graph.num_classes, spec_out=graph.num_classes,
                 conv1_in_dim=graph.num_node_features, conv1_out_dim=conv1_out,
                 conv2_out_dim=graph.num_classes,
                 num_linear=2, add_relu=True,
                 conv_bias=True,
                 pdrop=p_dropout).to(device)


    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type=mask_type,
                                num_epoch=num_epoch, data=graph, keepResult=train_keep, verbose=train_verbose)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=plt_keep, sh=plt_sh)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=plt_keep, sh=plt_sh)

    epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ########################################## end of training phase

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type=mask_type, keepResult=test_keep,
                      verbose=test_verbose)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    res = {'model_name':mdl_name, 'model':mdl, 'optimizer':opt,
           'epochResults':epochResults, 'summaryResults':summaryResults}
    #final_result = {str(mdl_name):res}

    ##########################################################################################  save the result
    try:
        final = pickle.load(open("dic.pickle", "rb"))
        final[str(mdl_name)] = res
    except (OSError, IOError) as e:
        final = {}
        final[str(mdl_name)] = res
    pickle.dump(final, open("dic.pickle", "wb"))


