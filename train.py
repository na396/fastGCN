
import time
from utils import *
import torch.nn.functional as F


######################################################################################################################### def train
def train(model, model_name, optimizer, num_epoch, data, mask_type="perClass", keepResult=True, verbose=True):

    # perform training phase on the input data
    # mdl: model
    # opt: optimizer
    # num_epoch: number of epoch
    # data: input semi-supervised graph

    if mask_type == "allClasses":
        data.train_mask_final = data.trainMask_allClasses
        data.val_mask_final = data.valMask_allClasses
        data.test_mask_final = data.testMask_allClasses
    elif mask_type == "original":
        data.train_mask_final = data.train_mask
        data.val_mask_final = data.val_mask
        data.test_mask_final = data.test_mask
    elif mask_type == "perClass":
        data.train_mask_final = data.trainMask_perClass
        data.val_mask_final = data.valMask_perClass
        data.test_mask_final = data.testMask_perClass

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

        loss_test = F.nll_loss(out[data.test_mask_final], data.y[data.test_mask_final])
        acc_test = accuracy(out[data.test_mask_final], data.y[data.test_mask_final])

        if keepResult:
            dic = {"model": model_name, "data": data.graph_name, "mask": mask_type,
                   "epoch_number": epoch+1, "epoch_time":time.time() - t,
                   "training_loss": loss_train.detach().numpy(), "training_accuracy": acc_train,
                   "validation_loss":loss_val.detach().numpy(), "validation_accuracy": acc_val,
                   "test_loss":loss_test.detach().numpy(), "test_accuracy":acc_test}

            detailDF = appendDF(detailDF, dic)

        if verbose:
           print( detailDF.loc[:, ["epoch_number", "epoch_time", "training_loss", "validation_loss", "test_loss",
                                   "training_accuracy", "validation_accuracy", "test_accuracy"] ].iloc[-1, :])
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