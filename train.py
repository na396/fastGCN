
import time
import warnings
from utils import *
import torch
import torch.nn.functional as F

######################################################################################################################### def trainTest_split
def trainValidationTest_split(data, train_percent=None, train_num=None, val_percent=None, val_num=None, verbose=True):

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

    if train_percent is not None:
        train_num = int(data.num_nodes * train_percent)

    ################################################################
    if (val_percent is not None) and (val_num is not None):
        warnings.warn('both val_percent and val_num are not None \n, making val_num to None...')
        val_num = None

    if (val_percent is None) and (val_num is None):
        raise TypeError("both percent and num cannot be None")

    if val_percent is not None:
        val_num = int(data.num_nodes * val_percent)

    ################################################################
    perm = torch.randperm(data.num_nodes)
    perm_train = perm[0:train_num]
    perm_val = perm[train_num:train_num+val_num]
    perm_test = perm[train_num+val_num:]

    indices = torch.arange(0, data.num_nodes)
    trainMask = torch.isin(indices, perm_train)
    valMask = torch.isin(indices, perm_val)
    testMask = torch.isin(indices, perm_test)

    data.trainMask = trainMask
    data.valMask = valMask
    data.testMask = testMask

    if verbose:
        print(data.graph_name)
        print("trainMask size: ", trainMask.size())
        print("trainMask.sum().item(): ", trainMask.sum().item())
        print("valMask size: ", valMask.size())
        print("valMask.sum().item(): ", valMask.sum().item())
        print("testMask size: ", testMask.size())
        print("testMask.sum().item(): ", testMask.sum().item())
        print("\n")

    return data

######################################################################################################################### def train
def train(model, model_name, optimizer, num_epoch, data, mask_type="manualMask", keepResult=True, verbose=True):

    # perform training phase on the input data
    # mdl: model
    # opt: optimizer
    # num_epoch: number of epoch
    # data: input semi-supervised graph

    if mask_type == "manualMask":
        data.train_mask_final = data.trainMask
        data.val_mask_final = data.valMask
        data.test_mask_final = data.testMask
    else:
        data.train_mask_final = data.train_mask
        data.val_mask_final = data.val_mask
        data.test_mask_final = data.test_mask

    if keepResult: detailDF = epochPerformanceDF()

    print("model: ", model_name)
    print("graph_name: ", data.graph_name)
    print("\n")

    t = time.time()
    model.train()
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        out = model(data)

        loss_train = F.nll_loss(out[data.train_mask_final], data.y[data.train_mask_final])
        acc_train = accuracy(out[data.train_mask_final], data.y[data.train_mask_final])

        loss_val = F.nll_loss(out[data.val_mask_final], data.y[data.val_mask_final])
        acc_val = accuracy(out[data.val_mask_final], data.y[data.val_mask_final])

        if keepResult:
            dic = {"model": model_name, "data": data.graph_name, "mask": mask_type,
                   "epoch_number": epoch+1, "epoch_time":time.time() - t,
                   "training_loss": loss_train.detach().numpy(), "training_accuracy": acc_train,
                   "validation_loss":loss_val.detach().numpy(), "validation_accuracy": acc_val}

            detailDF = appendDF(detailDF, dic)

        if verbose:
           print( detailDF.loc[:, ["epoch_number", "epoch_time", "training_loss", "validation_loss"]].iloc[-1, :])
           print("\n")

        loss_train.backward()
        optimizer.step()

    return model, optimizer, detailDF

######################################################################################################################### def test
def test(model, model_name, data, mask_type="manualMask", keepResult=True, verbose=True):

    sumDF = TrainValidationTestDF()

    if mask_type == "manualMask":
        data.train_mask_final = data.trainMask
        data.val_mask_final = data.valMask
        data.test_mask_final = data.testMask
    else:
        data.train_mask_final = data.train_mask
        data.val_mask_final = data.val_mask
        data.test_mask_final = data.test_mask


    model.eval()
    out = model(data)

    pred = model(data).argmax(dim=1)
    acc_model = (pred == data.y).sum()/len(data.y)

    loss_train = F.nll_loss(out[data.train_mask_final], data.y[data.train_mask_final])
    acc_train = accuracy(out[data.train_mask_final], data.y[data.train_mask_final])

    loss_val = F.nll_loss(out[data.val_mask_final], data.y[data.val_mask_final])
    acc_val = accuracy(out[data.val_mask_final], data.y[data.val_mask_final])

    loss_test = F.nll_loss(out[data.test_mask_final], data.y[data.test_mask_final])
    acc_test = accuracy(out[data.test_mask_final], data.y[data.test_mask_final])

    if keepResult:
        dic = {"model": model_name, "data": data.graph_name, "mask": mask_type, "model_accuracy": acc_model.item(),
               "training_loss": loss_train.detach().numpy(), "training_accuracy": acc_train,
               "validation_loss": loss_val.detach().numpy(), "validation_accuracy": acc_val,
               "test_loss": loss_test.detach().numpy(), "test_accuracy": acc_test}

        sumDF = appendDF(sumDF, dic)

    if verbose:
        print(sumDF.iloc[-1, :])
        print("\n")

    return model, sumDF