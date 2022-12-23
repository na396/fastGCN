import time
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

######################################################################################################################### class vanilaGCN
class vanilaGCN(torch.nn.Module):

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
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        x = self.conv_h2out(x, edge_index)

        return F.log_softmax(x, dim=1)

######################################################################################################################### class spectrumGCN_inLayer
class spectralGCN_inLayer(torch.nn.Module):

    # spectrum GCN for input layer
    # spectrum dimension

    def __init__(self, data, in_dim, h1_dim, h2_dim, out_dim, spec_out_dim, spec_in_dim=None,
                 allLayers=False, lin=False, bias=False, pdrop=0.5):

        super().__init__()

        self.spectra = data.eigenvectors[:, 0:spec_in_dim] if lin else data.eigenvectors[:, 0:spec_out_dim]
        self.allLayers = allLayers
        self.lin = lin
        self.bias = bias
        self.pdrop=pdrop

        ####################### methods
        if lin:
            self.lin_specInOut = Linear(spec_in_dim, spec_out_dim, bias=bias)
        in_dim = in_dim + spec_out_dim
        self.conv_inh1 = GCNConv(in_dim, h1_dim)

        if allLayers:
            h1_dim = h1_dim + spec_out_dim
        self.conv_h1h2 = GCNConv(h1_dim, h2_dim)

        if allLayers:
            h2_dim = h2_dim + spec_out_dim
        self.conv_h2out = GCNConv(h2_dim, out_dim)


    def forward(self, data):

        # an object of class data
        # determine nodes and edges

        x, edge_index = data.x, data.edge_index

        xp = self.lin_specInOut(self.spectra) if self.lin else self.spectra
        x = torch.cat((x, xp), 1)

        x = self.conv_inh1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        if self.allLayers: x = torch.cat((x, xp), 1)
        x = self.conv_h1h2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        if self.allLayers: x = torch.cat((x, xp), 1)
        x = self.conv_h2out(x, edge_index)

        return F.log_softmax(x, dim=1)

######################################################################################################################### class spectrumGCN_outLayer
class spectralGCN_outLayer(torch.nn.Module):

    # spectrum GCN for input layer
    # spectrum dimension

    def __init__(self, data, in_dim, h1_dim, h2_dim, out_dim, spec_out_dim, spec_in_dim=None,
                 lin=False, bias=False, pdrop=0.5):

        super().__init__()

        self.spectra = data.eigenvectors[:, 0:spec_in_dim] if lin else data.eigenvectors[:, 0:spec_out_dim]
        self.lin = lin
        self.bias = bias
        self.pdrop = pdrop

        ####################### methods
        self.conv_inh1 = GCNConv(in_dim, h1_dim)
        self.conv_h1h2 = GCNConv(h1_dim, h2_dim)

        if lin:
            self.lin_specInOut = Linear(spec_in_dim, spec_out_dim, bias=bias)
            h2_dim = h2_dim + spec_out_dim
        self.conv_h2out = GCNConv(h2_dim, out_dim)


    def forward(self, data):

        # an object of class data
        # determine nodes and edges

        x, edge_index = data.x, data.edge_index

        x = self.conv_inh1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        x = self.conv_h1h2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        if self.lin:
            xp = self.lin_specInOut(self.spectra) if self.lin else self.spectra
            x = torch.cat((x, xp), 1)

        x = self.conv_h2out(x, edge_index)

        return F.log_softmax(x, dim=1)

######################################################################################################################### class spectrumGCN_hiddenLayer
class spectralGCN_hiddenLayer(torch.nn.Module):

    # spectrum GCN for input layer
    # spectrum dimension

    def __init__(self, data, in_dim, h1_dim, h2_dim, out_dim, spec_out_dim, spec_in_dim=None,
                  allLayers=True, lin=False, bias=False, pdrop=0.5):

        super().__init__()

        self.spectra = data.eigenvectors[:, 0:spec_in_dim] if lin else data.eigenvectors[:, 0:spec_out_dim]
        self.lin = lin
        self.bias = bias
        self.allLayers = allLayers
        self.pdrop = pdrop

        ####################### methods

        self.conv_inh1 = GCNConv(in_dim, h1_dim)
        if self.lin:
            self.lin_specInOut = Linear(spec_in_dim, spec_out_dim, bias=bias)
        h1_dim = h1_dim + spec_out_dim
        self.conv_h1h2 = GCNConv(h1_dim, h2_dim)

        if allLayers:
            h2_dim = h2_dim + spec_out_dim
        self.conv_h2out = GCNConv(h2_dim, out_dim)


    def forward(self, data):

        # an object of class data
        # determine nodes and edges

        x, edge_index = data.x, data.edge_index

        x = self.conv_inh1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        xp = self.lin_specInOut(self.spectra) if self.lin else self.spectra
        x = torch.cat((x, xp), 1)
        x = self.conv_h1h2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.pdrop, training=self.training)

        if self.allLayers: x = torch.cat((x, xp), 1)
        x = self.conv_h2out(x, edge_index)

        return F.log_softmax(x, dim=1)

