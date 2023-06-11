
from layers import *
from torch.nn import Module

######################################################################################################################### class spectrumGCN
class spectrumMLP(Module):

    ######################################################### constructor
    def __init__(self, in_dim, hidden_dim, out_dim, num_lin=2, add_relu=True, bias=True, init="xavier", pdrop=0,
                 embedding="non-symmetric", deepwalk_epoch=5000, deepwalk_lr=0.01, deepwalk_maskType="original",
                 deepwalk_batchSize=128, cache=False):

        super().__init__()

        ####################### attributes

        self.p_dropout = pdrop # probability for dropoup out
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_lin = num_lin
        self.add_relu = add_relu
        self.bias = bias
        self.init = init
        self.embedding = embedding
        self.deepwalk_epoch = deepwalk_epoch
        self.deepwalk_lr = deepwalk_lr
        self.deepwalk_batchSize = deepwalk_batchSize
        self.deepwalk_maskType = deepwalk_maskType
        self.cache = cache

        # calculates
        self.model_name = f"spectrumMLP + num_linear={self.num_lin} + " \
                          f"add_relu={self.add_relu} + init={init} +  dropout={self.p_dropout} + " \
                          f"embedding={self.embedding} + in_dim={self.in_dim}"

        ####################### layers
        self.linears = dynamicMLP(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                   num_lin=self.num_lin, add_relu=self.add_relu, bias=self.bias, init=self.init)

    ######################################################### __str__
    def __str__(self):
        settings = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        self.model_name = f"spectrumGCN({settings})"
        return f"spectrumMLP({settings})"

    ######################################################### forward
    def forward(self, data):

        x = data.embedding_vectors[:, 0:self.in_dim]
        x = self.linears(x) # a tensor of n * 2k
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        return F.log_softmax(x, dim=1)
