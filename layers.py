import warnings
import torch

from scipy.linalg import fractional_matrix_power

import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import Module
from torch.nn import ModuleList

from torch.nn.parameter import Parameter

from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree


######################################################################################################################### spectral_concatenation
class spectral_concatenation(Module):

    # performs the concatenation
    def __init__(self, spec_in, spec_out, num_linear=2, add_relu=True):

        # spec_in: input dimension for each linear layer
        # spec_out: output dimension for each liner layer
        # num_linear: number of linear layer, default=2
        # add_relu: if True, will add relu after normalization

        super().__init__()

        if num_linear ==1:
            warnings.warn('number of linear layers for the spectral concatenation is 1, are you sure!...')
        if num_linear ==0:
            raise TypeError("number of linear layers for spectral concatenation is 0 \n it must be greater than 0")

        self.numlinear = num_linear
        self.add_relu = add_relu
        self.linears = ModuleList([Linear(in_features=spec_in, out_features=spec_out, bias=False) for _ in range(num_linear)])

    def row_normalize(self, X):
        return F.normalize(input=X, p=2.0, dim=1, eps=1e-12, out=None)

    def forward(self, X):
        # ModuleList can act as an iterable, or be indexed using ints

        embeds = []
        for i, lin in enumerate(self.linears):
            Y = lin(X)
            Y = self.row_normalize(Y)
            if self.add_relu:
                Y = F.relu(Y)
            embeds.append(Y)
        out = torch.cat(tensors=[y for i, y in enumerate(embeds)], dim=1, out=None)
        return out

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
    def forward(self, x, edge_index, eg_vectors):
        # performs graph convolution neural networks
        # it uses non-symmetric graph laplacian

        # data: an object of data in torch_geometric
        #      x: feature map
        #      edge_index: edge_index
        #      eg_vectors = reduced eg_vectors

        if self.graph_less: # graph convGCN with creating the adjacency matrix

            u = eg_vectors.sum(dim=0)
            deg = eg_vectors @ u
            deg = 1./deg
            eg_vectors_T = torch.transpose(eg_vectors, 0, 1)
            out = torch.diag(deg) @ eg_vectors @ (eg_vectors_T @ x)
            out = out @ self.weights
            if self.bias is not None:
                return out + self.bias

        else: # we have the graph
            deg = degree(edge_index[0])
            deg = deg+1 # add_self_loops is always True
            x = torch.from_numpy(fractional_matrix_power(torch.diag(deg), (1/2))) @ x
            x = self.conv(x, edge_index)
            out = torch.from_numpy(fractional_matrix_power(torch.diag(deg), (-1/2))) @ x
        return out

    ############################ __repr__
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ', ' + \
               'graph_less=' + str(self.graph_less) + ', ' + \
               'bias=' + str(True if self.bias is not None else False) + ')'