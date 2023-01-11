

from layers import *

from torch.nn import Linear
from torch.nn import Module

from torch_geometric.nn import GCNConv

######################################################################################################################### class spectrumGCN_inLayer
class spectralGCN_inLayer(Module):

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

######################################################################################################################### main function

######################################################################################################################### model: spectrumGCN_outLayer

mdl_name = "spectrumGCN_outLayer"
for key in graphs_name:
    ##################################################### initialization
    graph = graphs[key]
    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = spectralGCN_outLayer(data=graph,
                      in_dim=graph.num_node_features,
                      h1_dim=h1_dim,
                      h2_dim=h2_dim,
                      out_dim=graph.num_classes,
                      spec_in_dim=graph.num_classes,
                      spec_out_dim=2*graph.num_classes,
                      lin=True, bias=False, pdrop=pdrop).to(device)

    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type="manualMask",
                                    num_epoch=num_epoch, data=graph, keepResult=True, verbose=False)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=True, sh=False)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=True, sh=False)

    results = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type="manualMask", keepResult=True,
                      verbose=True)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    modelResuls[mdl_name] = mdl
    optimizerResults[mdl_name] = opt

######################################################################################################################### model: spectrumGCN_hiddenLayer

mdl_name = "spectrumGCN_hiddenLayer"
for key in graphs_name:
    ##################################################### initialization
    graph = graphs[key]
    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    mdl = spectralGCN_hiddenLayer(data=graph,
                      in_dim=graph.num_node_features,
                      h1_dim=h1_dim,
                      h2_dim=h2_dim,
                      out_dim=graph.num_classes,
                      spec_in_dim=graph.num_classes,
                      spec_out_dim=2*graph.num_classes,
                      lin=True, bias=False, allLayers=False, pdrop=pdrop).to(device)

    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)

    print("\n")
    ##################################################### training phase
    print("entring training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type="manualMask",
                                    num_epoch=num_epoch, data=graph, keepResult=True, verbose=False)

    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="loss", keep=True, sh=False)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, col="accuracy", keep=True, sh=False)

    results = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ##################################################### test phase
    print("entring test phase...\n")
    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type="manualMask", keepResult=True,
                      verbose=True)
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ##################################################### saving results
    print("saving results...")
    modelResuls[mdl_name] = mdl
    optimizerResults[mdl_name] = opt