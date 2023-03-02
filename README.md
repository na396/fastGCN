# fastGCN
fastGCN is a novel design for graph convolutional neural network that utilizes the spectral embedding of the adjacency matrix.

## data.py includes function for data preprocessing
#### data_summary, data_prepare, data_cleaning will check the graph, 
#### it makes graph undirected, without loop.
#### it adds the eigenvectors and eigenvalues to the graph, data object as data.eigevectors, and data.eigenvalues
#### it also adds two type of masking in addition to the original masking; i) "perClass": which consider the number of datapoints per class, ii) "allClasses": which regardless of the number of datapoint per class, it randomly picks train, validation and test sets.
####    iii) original: refers to original masking of the graph.
  
 ## train.py includes function train and test, for the train and test phase.
 ### each function return a dataframe in which the information for each epoch is stored. Additionally, it plots the loss value and accuracy of the model at each epoch.
 
 ## spectrumGCN.py includes the main class for fastGCN.
 ### it runs in two modes: i) with graph and ii) graph less
 ### graph less is the mode in which we created our adjacency matrix by inner poridcut over the eigenvectors. Inside this class, other classes are called; i)s pectral_concatenation, ii) convGCN
 ### these classes are implemented in layers.py
 
## layers.py includes spectral_concatenation, and convGCN
### spectral_concatenation: performs the eigenvector concatenation with n linear layer (default: n=2), it uses parallel computing.
### convGCN: is our convolutional graph neural networks. It runs in two modes, i) whith graph and ii) wihtout graph (or graphless)
#### with gaph mode is the same as vanila GCNconv of pytorch. We only slightly changed this layer so it works with non-symmetric graph Laplacian.
#### in graphless or without graph mode, a new layer is implemented with sctrach. It does not utilize any property of graph sparsity as our graph is dense.

## utils includes practical functions. like 
#### i) calculating the accuracy at a given step, 
#### ii) plotting the accuracy and loss per epoch in a model. 
#### iii) showng the current status of pytorch with regard to GPU, version, etc.
#### iv) creating a pandas dataframe to store the detailed information per epoch.



