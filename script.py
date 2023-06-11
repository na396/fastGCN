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

from paramteres_spectrumGCN import *

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


######################################################################################################################### import main
import main