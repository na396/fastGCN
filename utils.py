"""
Practical Fucntions
"""
import pandas as pd
import matplotlib.pyplot as plt
import torch

######################################################################################################################### def torchStatus
def torchStatus():

    print("torch_version: ", torch.__version__)
    print("torch CUDA version: ", torch.version.cuda)
    print("torch CUDA available: ", torch.cuda.is_available())
    print("torch number of GPU: ", torch.cuda.device_count())


######################################################################################################################### def TrainValidationTestDF
def TrainValidationTestDF():

    # creates a dataframe showing the model accuracy

    df = pd.DataFrame(columns = ["model", "data", "mask", "model_accuracy",
                                "training_loss", "validation_loss", "test_loss",
                                 "training_accuracy", "validation_accuracy", "test_accuracy"])
    return df

######################################################################################################################### def epochPerformanceDF
def epochPerformanceDF():

    # creates a dataframe showing the model accuracy

    df = pd.DataFrame( columns = ["model", "data", "mask",
                                  "epoch_number", "epoch_time",
                                  "training_loss", "validation_loss", "test_loss",
                                  "training_accuracy", "validation_accuracy", "test_accuracy" ])
    return df

######################################################################################################################### def addToDF
def appendDF(df, dic):

    inds = len(df.index)
    for key in dic.keys():
        df.loc[inds, key] = dic[key]
    return df

######################################################################################################################### def accuracy
def accuracy(pred, true_label, verbose=False):

    # mdl: model
    # data: graph
    # inds: index for either train, validation, or test
    pred = pred.argmax(dim=1)
    correct = (pred == true_label).sum()
    acc = int(correct) / len(true_label)

    if verbose: print(f'Accuracy: {acc:.4f}')
    return acc

######################################################################################################################### def plot_epoch
def plot_epoch(df, model_name, data_name, plt_dir, graph_mode, col = "loss", keep=True, sh=True, imagetype="png"):

    # plots the train/validation loss with respect to number of epoch
    # df.column:
    # "model": model name, either vanilaGCN,
    # "Data": grph name,
    # "epoch_number": epoch number
    # "epoch_time": running time for the epoch
    # "training_loss": loss value for the train at epoch_number
    # "training_accuracy": accuracy for the train at epoch_number
    # "validation_loss": loss value for the validation at epoch_number
    # "validation_accuracy":accuracy for the validation at epoch_number

    if col == "loss": cols=["training_loss", "validation_loss", "test_loss"]

    elif col == "accuracy": cols=["training_accuracy", "validation_accuracy", "test_accuracy"]
    else: raise ValueError('col in plot_epoch is not defines...')

    matn = "WithOutGraph" if graph_mode == False else "WithGraph"
    cap = str(model_name) + "_" + str(data_name) + "_" + str(matn) + "_Training and Validation " + str(col)
    plt.figure()
    plt.plot(df["epoch_number"], df[cols[0]], color='b', label='training set')
    plt.plot(df["epoch_number"], df[cols[1]], color='g', label='validation set')
    plt.plot(df["epoch_number"], df[cols[2]], color='r', label='test set')
    plt.title(cap)
    plt.xlabel("epoch")
    plt.ylabel(col)
    plt.legend()

    if keep:
        cap = str(plt_dir) + str(model_name) + "_ " + str(data_name)+ "_" + str(col) + "_TrainAndValidation."+str(imagetype)
        plt.savefig(cap)
    if sh: plt.show()
    plt.close()

