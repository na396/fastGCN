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

######################################################################################################################### class spectrumGCN (the main class)
class spectrumGCN(Module):

    ######################################################### constructor
    def __init__(self, graph_less,
                 spec_in, spec_out,
                 conv1_in_dim, conv1_out_dim, conv2_out_dim,
                 num_linear=2, add_relu=True,
                 conv_bias: bool=True, multiple_concatenations=False,
                 pdrop=0.5, eg_features=False, eg_features_dim=None):
        super().__init__()

        ####################### attributes

        self.graph_less = graph_less  # model mode
        self.p_dropout = pdrop # probability for dropoup out
        self.spec_in = spec_in # number of selected dimension for the input of spectral_concatenation class, default = 2*num_classes
        self.spec_out = spec_out  # number of selected dimension for the output of  spectral_concatenation class, default = num_classes
        self.conv1_in_dim = conv1_in_dim
        self.conv1_out_dim = conv1_out_dim # output dimension for convGCN layer 1
        self.conv2_out_dim = conv2_out_dim # output dimension for convGCN layer 2, number of classes
        self.num_linear = num_linear # number of linear layer for spectral_concatenation, default=2
        self.add_relu = add_relu # add_relu: adding relu layer in spectral_concatenation, default=True
        self.add_self_loops = True # always add self_loops
        self.normalize = True # always normalize
        self.conv_bias = conv_bias # if convGCN layer has bias, default=True
        self.eg_features = eg_features
        self.multiple_concatenations = multiple_concatenations
        self.eg_features_dim = eg_features_dim

        # calculates
        self.conv2_in_dim = conv1_out_dim
        self.num_node = self.eg_features_dim if self.eg_features else self.conv1_in_dim

        parameters = self.get_parameters()
        self.model_name = " + ".join(f"{key}={value}" for key, value in parameters.items())

        ####################### layers
        self.spec_concat_1 = spectral_concatenation(spec_in=spec_in, spec_out=spec_out,
                                                  num_linear=num_linear, add_relu=add_relu)

        self.spec_concat_2 = spectral_concatenation(spec_in=spec_in, spec_out=spec_out,
                                                    num_linear=num_linear, add_relu=add_relu)

        self.conv_1 = convGCN(graph_less=graph_less, in_dim=self.num_node, out_dim=conv1_out_dim,
                              have_bias=conv_bias)
        self.conv_2 = convGCN(graph_less=graph_less, in_dim=self.conv2_in_dim, out_dim=conv2_out_dim,
                              have_bias=conv_bias)
    ######################################################### get_parameters
    def get_parameters(self):
        parameters = {
            'graph_less': self.graph_less,
            'pdrop': self.p_dropout,
            'spec_in': self.spec_in,
            'spec_out': self.spec_out,
            'conv1_in_dim': self.conv1_in_dim,
            'conv1_out_dim': self.conv1_out_dim,
            'conv2_out_dim': self.conv2_out_dim,
            'num_linear': self.num_linear,
            'add_relu': self.add_relu,
            'add_self_loops': self.add_self_loops,
            'normalize': self.normalize,
            'conv_bias': self.conv_bias,
            'eg_features': self.eg_features,
            'multiple_concatenations': self.multiple_concatenations,
            'eg_features_dim': self.eg_features_dim,
            'conv2_in_dim': self.conv2_in_dim,
            'num_node': self.num_node
        }
        return parameters

    ######################################################### __str__
    def __str__(self):
        settings = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"spectrumMLP({settings})"

    ######################################################### forward
    def forward(self, data):

        # determine nodes and edges
        x, edge_index, eg_vectors = data.x, data.edge_index, data.eigenvectors[:, 0:self.spec_in]
        if self.eg_features: x = data.eigenvectors[:, 0:self.eg_features_dim]

        if self.graph_less:

            # eg_vecotrs is the processed spectral concatenation, i.e. Xp
            Xp1 = self.spec_concat_1(eg_vectors) # a tensor of n * 2k

            x = self.conv_1(x, edge_index, Xp1) # a tensor of n * conv1_out_dim
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

            Xp2 = self.spec_concat_2(eg_vectors) if self.multiple_concatenations else eg_vectors # a tensor of n * 2k
            x = self.conv_2(x, edge_index, Xp2)

            return F.log_softmax(x, dim=1)

        else:
            x = self.conv_1(x, edge_index, eg_vectors)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

            x = self.conv_2(x, edge_index, eg_vectors)

            return F.log_softmax(x, dim=1)

######################################################################################################################### class spectrumGCN_multipleConcat
class spectrumGCN_multipleConcat(Module):

    def __init__(self, graph_less,
                 spec_in, spec_out,
                 conv1_in_dim, conv1_out_dim,
                 conv2_out_dim,
                 num_linear=2, add_relu=True,
                 conv_bias: bool=True, multiple_concatenations=False,
                 pdrop=0.5, eg_features=False, eg_features_dim=None):
        super().__init__()

        ####################### attributes
        self.graph_less = graph_less  # model mode
        self.p_dropout = pdrop # probability for dropoup out
        self.spec_in = spec_in # number of selected dimension for the input of spectral_concatenation class, default = 2*num_classes
        self.spec_out = spec_out # number of selected dimension for the output of  spectral_concatenation class, default = num_classes
        self.conv1_in_dim = conv1_in_dim
        self.conv1_out_dim = conv1_out_dim # output dimension for convGCN layer 1
        self.conv2_out_dim = conv2_out_dim # output dimension for convGCN layer 2, number of classes
        self.num_Linear = num_linear # number of linear layer for spectral_concatenation, default=2
        self.add_relu = add_relu # add_relu: adding relu layer in spectral_concatenation, default=True
        self.add_self_loops = True # always add self_loops
        self.normalize = True # always normalize
        self.conv_bias = conv_bias # if convGCN layer has bias, default=True
        self.eg_features = eg_features
        self.multiple_concatenations = multiple_concatenations
        self.eg_features_dim = eg_features_dim

        # calculates
        self.conv2_in_dim = conv1_out_dim

        temp1 = "multiple concatenation" if self.multiple_concatenations else "single concatenation"
        temp2 = "eigenvectors as features " if self.eg_features else "features as features"
        temp3 = "WithOutGraph" if self.graph_less else 'WithGraph'

        self.model_name = "spectrumGCN" + " + " + str(temp3) + " + " + str(temp1) + " + " + str(temp2) + " + dropout=" + str(self.p_dropout)
        self.num_node = self.eg_features_dim if self.eg_features else self.conv1_out_dim

        ####################### layers
        self.spec_concat_1 = spectral_concatenation(spec_in=spec_in, spec_out=spec_out,
                                                  num_linear=num_linear, add_relu=add_relu)

        self.spec_concat_2 = spectral_concatenation(spec_in=spec_in, spec_out=spec_out,
                                                    num_linear=num_linear, add_relu=add_relu)

        self.conv_1 = convGCN(graph_less=graph_less, in_dim=self.num_node, out_dim=conv1_out_dim,
                              have_bias=conv_bias)
        self.conv_2 = convGCN(graph_less=graph_less, in_dim=self.conv2_in_dim, out_dim=conv2_out_dim,
                              have_bias=conv_bias)

    def forward(self, data):

        # determine nodes and edges
        x, edge_index, eg_vectors = data.x, data.edge_index, data.eigenvectors[:, 0:self.spec_in]
        if self.eg_features: x = data.eigenvectors[:, 0:self.spec_in]

        if self.graph_less:

            # eg_vecotrs is the processed spectral concatenation, i.e. Xp
            Xp1 = self.spec_concat_1(eg_vectors) # a tensor of n * 2k

            x = self.conv_1(x, edge_index, Xp1) # a tensor of n * conv1_out_dim
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

            Xp2 = self.spec_concat_2(eg_vectors) if self.multiple_concatenations else eg_vectors # a tensor of n * 2k
            x = self.conv_2(x, edge_index, Xp2)

            return F.log_softmax(x, dim=1)

        else:
            x = self.conv_1(x, edge_index, eg_vectors)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

            x = self.conv_2(x, edge_index, eg_vectors)

            return F.log_softmax(x, dim=1)