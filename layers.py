import warnings
import torch

from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree


######################################################################################################################### spectral_concatenation
class spectral_concatenation(Module):

    # performs the concatenation
    def __init__(self, spec_in, spec_out, num_linear=2, add_relu=True):

        # spec_in: input dimension for each linear layer, and also the number of dimension of egenvectors
        # spec_out: output dimension for each liner layer
        # num_linear: number of linear layer, default=2
        # add_relu: if True, will add relu after normalization

        super().__init__()

        if num_linear ==1:
            warnings.warn('number of linear layers for the spectral concatenation is 1, are you sure!...')
        if num_linear ==0:
            raise TypeError("number of linear layers for spectral concatenation is 0 \n it must be greater than 0")

        self.spec_in = spec_in
        self.spec_out = spec_out
        self.numlinear = num_linear
        self.add_relu = add_relu
        self.reduce = torch.nn.Parameter(torch.FloatTensor(num_linear, spec_in, spec_out))

        self.reset_parameters()
    ############################ reset_parameters
    def reset_parameters(self):
        # performs xavier weight initialization
        #torch.nn.init.xavier_uniform_(self.reduce)
        for nl in range(self.numlinear):
            torch.nn.init.orthogonal_(self.reduce[nl])

    ############################ row_normalize
    def row_normalize(self, X):
        # normalize row of X to 1
        return F.normalize(input=X, p=2.0, dim=2, eps=1e-12, out=None)

    ############################ forward
    def forward(self, X):

        # ModuleList can act as an iterable, or be indexed using ints
        # X: a matric spec_in time spec_out

        y = X @ self.reduce # X: eigenvectors, has spec_in=2k dimension, self.reduce [num_linear, spec_in=2k, specout=k]
        y = self.row_normalize(y)
        y = torch.transpose(y, 0, 1)
        y = torch.flatten(y, start_dim=1)
        if self.add_relu:
            y = F.relu(y)
        return y

    ############################ __repr__
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.spec_in) + ' -> ' \
               + str(self.spec_out) + ', ' + \
               'num_linear=' + str(self.numlinear) + ', ' + \
               'add_relu=' + str(self.add_relu) + ')'

######################################################################################################################### convGCN
class convGCN(Module):

    def __init__(self, graph_less: bool, in_dim: int, out_dim: int, have_bias : bool=True):

        super().__init__()

        ####################### attributes & methods
        self.graph_less = graph_less
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.add_self_loops = True
        self.normalize = True

        if graph_less:
            self.weights = Parameter(torch.FloatTensor(in_dim, out_dim))
            if have_bias:
                self.bias = Parameter(torch.FloatTensor(out_dim))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        else:
            self.conv = GCNConv(in_channels=in_dim, out_channels=out_dim, bias=have_bias)

    ############################ reset_parameters
    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    ############################ forward
    def forward(self, x, edge_index, Xp):
        # performs graph convolution neural networks
        # it uses non-symmetric graph laplacian

        # data: an object of data in torch_geometric
        #      x: feature map
        #      edge_index: edge_index
        #     Xp: reduced eg_vectors

        if self.graph_less: # graph convGCN with creating the adjacency matrix
            u = Xp.sum(dim=0)
            deg = Xp @ u
            deg = 1./deg
            eg_vectors_T = torch.transpose(Xp, 0, 1)
            out = torch.diag(deg) @ Xp @ (eg_vectors_T @ x)
            out = out @ self.weights
            if self.bias is not None:
                out + self.bias

        else: # we have the graph
            deg = degree(edge_index[0])
            deg = deg+1 # add_self_loops is always True
            deg_12 = torch.pow(deg, 1/2)
            out = torch.diag(deg_12) @ x
            out= self.conv(out, edge_index)
            deg_12 = torch.pow(deg, -1/2)
            out = torch.diag(deg_12) @ out
        return out

    ############################ __repr__
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ', ' + \
               'graph_less=' + str(self.graph_less) + ', ' + \
               'bias=' + str(True if self.bias or self.bias is not None else False) + ')'