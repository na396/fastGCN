
from layers import *

from torch.nn import Module

from torch_geometric.nn import GCNConv

######################################################################################################################### class vanilaGCN2
class vanilaGCN(Module):

    def __init__(self, in_dim, h1_dim, h2_dim, out_dim, pdrop=0.5):
        super().__init__()

        ####################### methods
        self.conv_inh1 = GCNConv(in_dim, h1_dim)
        self.conv_h1h2 = GCNConv(h1_dim, h2_dim)
        self.conv_h2out = GCNConv(h2_dim, out_dim)
        self.pdrop=pdrop

    def forward(self, data):
        # an object of class data
        # determine nodes and edges
        x, edge_index = data.x, data.edge_index

        x = self.conv_inh1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        x = self.conv_h1h2(x, edge_index)

        return F.log_softmax(x, dim=1)

######################################################################################################################### class spectraGCN
class spectrumGCN(Module):

    def __init__(self, graph_less,
                 spec_in, spec_out,
                 conv1_in_dim, conv1_out_dim,
                 conv2_out_dim,
                 num_linear=2, add_relu=True,
                 conv_bias: bool=True,
                 pdrop=0.5):
        super().__init__()

        ####################### attributes
        self.graph_less = graph_less  # model mode
        self.p_dropout = pdrop # probability for dropoup out
        self.spec_in = spec_in # number of selected dimension for the input of spectral_concatenation class, default = 2*num_classes
        self.spec_out = spec_out  # number of selected dimension for the output of  spectral_concatenation class, default = num_classes
        self.conv1_in_dim = conv1_in_dim
        self.conv1_out_dim = conv1_out_dim # output dimension for convGCN layer 1
        self.conv2_out_dim = conv2_out_dim # output dimension for convGCN layer 2, number of classes
        self.num_Linear = num_linear # number of linear layer for spectral_concatenation, default=2
        self.add_relu = add_relu # add_relu: adding relu layer in spectral_concatenation, default=True
        self.add_self_loops = True # always add self_loops
        self.normalize = True # always normalize
        self.conv_bias = conv_bias # if convGCN layer has bias, default=True

        # calculates
        self.conv2_in_dim = conv1_out_dim

        ####################### layers
        self.spec_concat = spectral_concatenation(spec_in=spec_in, spec_out=spec_out,
                                                  num_linear=num_linear, add_relu=add_relu)
        self.conv_1 = convGCN(graph_less=graph_less, in_dim=conv1_in_dim, out_dim=conv1_out_dim,
                              have_bias=conv_bias)
        self.conv_2 = convGCN(graph_less=graph_less, in_dim=self.conv2_in_dim, out_dim=conv2_out_dim,
                              have_bias=conv_bias)

    def forward(self, data):

        # determine nodes and edges
        x, edge_index, eg_vectors = data.x, data.edge_index, data.eigenvectors[:, 0:self.spec_in]

        if self.graph_less:

            eg_vectors = self.spec_concat(eg_vectors) # a tensor of n * 2k

            x = self.conv_1(x, edge_index, eg_vectors) # a tensor of n * conv1_out_dim
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

            x = self.conv_2(x, edge_index, eg_vectors)

            return F.log_softmax(x, dim=1)

        else:
            x = self.conv_1(x, edge_index, eg_vectors)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

            x = self.conv_2(x, edge_index, eg_vectors)

            return F.log_softmax(x, dim=1)

