"""
Practical Functions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

######################################################################################################################### def check_directory
def check_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")

######################################################################################################################### def torchStatus
def torchStatus():

    print("torch_version: ", torch.__version__)
    print("torch CUDA version: ", torch.version.cuda)
    print("torch CUDA available: ", torch.cuda.is_available())
    print("torch number of GPU: ", torch.cuda.device_count())

######################################################################################################################### def gpu_setup
"""
    GPU Setup
"""
def gpu_setup(use_gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

######################################################################################################################### def TrainValidationTestDF
def TrainValidationTestDF():

    # creates a dataframe showing the model accuracy

    df = pd.DataFrame(columns = ["model", "data", "mask", "model_accuracy",
                                "training_loss", "validation_loss", "test_loss",
                                 "training_accuracy", "validation_accuracy", "test_accuracy", "iteration_number"])
    return df

######################################################################################################################### def epochPerformanceDF
def epochPerformanceDF():

    # creates a dataframe showing the model accuracy

    df = pd.DataFrame(columns = ["model", "data", "mask",
                                  "epoch_number", "epoch_time",
                                  "training_loss", "validation_loss", "test_loss",
                                  "training_accuracy", "validation_accuracy", "test_accuracy", "iteration_number"])
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
def plot_epoch(df, graph_name, root_dir, sc_num, col = "loss", keep=True, sh=True, imagetype="png",
               iter_num=None):

    # plots the train/validation loss with respect to number of epoch

    # df.column:
    #   "model": model name, either vanilaGCN,
    #   "Data": grph name,
    #   "epoch_number": epoch number
    #   "epoch_time": running time for the epoch
    #   "training_loss": loss value for the train at epoch_number
    #   "training_accuracy": accuracy for the train at epoch_number
    #   "validation_loss": loss value for the validation at epoch_number
    #   "validation_accuracy":accuracy for the validation at epoch_number

    if col == "loss":
        cols = ["training_loss", "validation_loss", "test_loss"]
        train_opt = df["training_loss"].min()
        pos_train = np.where(df["training_loss"] == train_opt)[0][0]
        val_opt = df["validation_loss"].min()
        pos_val = np.where(df["validation_loss"] == val_opt)[0][0]
        test_opt = df["test_loss"].min()
        pos_test = np.where(df["test_loss"] == test_opt)[0][0]

    elif col == "accuracy":
        cols = ["training_accuracy", "validation_accuracy", "test_accuracy"]
        train_opt = df["training_accuracy"].min()
        pos_train = np.where(df["training_accuracy"] == train_opt)[0][0]
        val_opt = df["validation_accuracy"].min()
        pos_val = np.where(df["validation_accuracy"] == val_opt)[0][0]
        test_opt = df["test_accuracy"].min()
        pos_test = np.where(df["test_accuracy"] == test_opt)[0][0]

    else : raise ValueError('col in plot_epoch is not defines...')


    plt.figure()
    # ax = fig.add_subplot(111)
    # ax.annotate('best train', xy=(pos_train, pos_train), xytext=(pos_train, (pos_train+1,)))
    # ax.annotate('best val', xy=(pos_val, val_opt), xytext=(pos_val, (val_opt + 1,)))
    # ax.annotate('best test', xy=(pos_test, test_opt), xytext=(pos_test, (test_opt + 1,)))

    plt.plot(df["epoch_number"], df[cols[0]], color='b', label='training set')
    plt.plot(df["epoch_number"], df[cols[1]], color='g', label='validation set')
    plt.plot(df["epoch_number"], df[cols[2]], color='r', label='test set')
    plt.title(f"Script{sc_num}_{col}")
    plt.xlabel("epoch")
    plt.ylabel(col)
    plt.legend()

    if keep:
        if iter_num != None:
            check_directory(f"{root_dir}/Results/Script{sc_num}/{graph_name}/{iter_num}")
            cap = f"{root_dir}/Results/Script{sc_num}/{graph_name}/{iter_num}/{graph_name}_{iter_num}_{col}.{imagetype}"
        else:
            check_directory(f"{root_dir}/Results/Script{sc_num}/")
            cap = f"{root_dir}/Results/Script{sc_num}/{graph_name}/{graph_name}_{col}.{imagetype}"
        plt.savefig(cap)
    if sh: plt.show()
    plt.close()

######################################################################################################################### def plot_epoch
