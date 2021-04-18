import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

EPS = 1e-15
MAX_LOGVAR = 10
edge_type = 4
random.seed(1234)


def negative_sampling(pos_edge_index, num_nodes):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0)


class OuterProductDecoder(torch.nn.Module):
    def __init__(self, node_hidden_dim):
        super(OuterProductDecoder, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.fc = nn.Linear(node_hidden_dim ** 2, edge_type)

    def forward(self, z, edge_index):
        z_in = torch.bmm(z[edge_index[0]].unsqueeze(2), z[edge_index[1]].unsqueeze(1)).view(-1, z.size(1) ** 2)
        value = self.fc(z_in)
        return value


class AutoEncoder(torch.nn.Module):
    def __init__(self, node_hidden_dim, encoder, decoder=None):
        super(AutoEncoder, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.encoder = encoder
        self.decoder = OuterProductDecoder(self.node_hidden_dim) if decoder is None else decoder
        AutoEncoder.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test :obj:`torch_geometric.data.Data` objects.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data
        # negative sampling
        neg_edge_index = negative_sampling(data.edge_index, data.x.size(0))
        edges = torch.cat([data.edge_index, neg_edge_index], dim=1)
        attr = torch.cat([data.edge_attr, data.edge_attr.new_zeros(neg_edge_index.size(1), edge_type)], dim=0)

        row, col = edges

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        perm = torch.randperm(row.size(0))
        row, col, attr = row[perm], col[perm], attr[perm]

        r, c, a = row[:n_v], col[:n_v], attr[:n_v]
        data.val_edge_index = torch.stack([r, c], dim=0)
        data.val_edge_attr = a
        val_data = Data(
            x=data.x,
            edge_index=data.val_edge_index,
            edge_attr=data.val_edge_attr)

        r, c, a = row[n_v:n_v + n_t], col[n_v:n_v + n_t], attr[n_v:n_v + n_t]
        data.test_edge_index = torch.stack([r, c], dim=0)
        data.test_edge_attr = a
        ts_data = Data(
            x=data.x,
            edge_index=data.test_edge_index,
            edge_attr=data.test_edge_attr)

        r, c, a = row[n_v + n_t:], col[n_v + n_t:], attr[n_v + n_t:]
        data.train_edge_index = torch.stack([r, c], dim=0)
        data.train_edge_attr = a
        tr_data = Data(
            x=data.x,
            edge_index=data.train_edge_index,
            edge_attr=data.train_edge_attr)

        return tr_data, val_data, ts_data

    def shuffle(self, data):
        perm = torch.randperm(data.train_pos_edge_index.size(1))
        data.train_pos_edge_index = data.train_edge_index[perm]
        data.train_pos_edge_attr = data.train_edge_attr[perm]
        return data

    def get_batch(self, data, i, bs):
        return data.train_pos_edge_index[:, i*bs:(i+1)*bs], data.train_pos_edge_attr[i*bs:(i+1)*bs, :]

    def log_loss(self, z, edge_index, edge_attr):
        loss = F.binary_cross_entropy_with_logits(
            self.decoder(z, edge_index), edge_attr)
        return loss

    def test(self, z, edge_index, edge_attr):
        y = edge_attr
        pred = self.decoder(z, edge_index)
        pred_c = pred > 0 # predict class
        y, pred, pred_c = y.detach().cpu().numpy(), pred.detach().cpu().numpy(), pred_c.detach().cpu().numpy()

        return roc_auc_score(y, pred, average="weighted"), precision_score(
            y, pred_c, average="weighted"), recall_score(y, pred_c, average="weighted"), accuracy_score(y, pred_c)

