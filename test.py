
import torch
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch import DeepWalk
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

dataset = CoraGraphDataset()
g1 = dataset[0]
model = DeepWalk(g1)
dataloader1 = DataLoader(torch.arange(g1.num_nodes()), batch_size=128,
                        shuffle=True, collate_fn=model.sample)
optimizer1 = SparseAdam(model.parameters(), lr=0.01)
num_epochs = 2

flag = True
for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    for batch_walk in dataloader1:
        if flag:
            nil = batch_walk
            flag=False
        print(f"batch_walk \n {batch_walk}")
        loss = model(batch_walk)
        print(f"loss {loss}")
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

train_mask = g1.ndata['train_mask']
test_mask = g1.ndata['test_mask']
X = model.node_embed.weight.detach()
y = g1.ndata['label']
clf = LogisticRegression().fit(X[train_mask].numpy(), y[train_mask].numpy())
clf.score(X[test_mask].numpy(), y[test_mask].numpy())