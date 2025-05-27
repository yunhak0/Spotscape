import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    Linear, BatchNorm, LayerNorm, GCNConv, GATConv, GraphConv, GINEConv, Sequential
)
from copy import copy


class AutoEncoder_Encoder(nn.Module):
    def __init__(self, input_dim, layer_sizes, enc_type,
                 act_fn=nn.ReLU(), drop_rate=0.0, batchnorm_mm=0.99):
        """AutoEncoder Linear Encoder Block"""
        super(AutoEncoder_Encoder, self).__init__()
        self.enc_type = enc_type

        hid_dims = [input_dim] + layer_sizes

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hid_dims[:-1], hid_dims[1:])):
            if enc_type.lower() == 'linear':
                layers.append(nn.Dropout(drop_rate))
                layers.append(Linear(in_dim, out_dim))
                layers.append(nn.BatchNorm1d(out_dim, momentum=batchnorm_mm))
            elif enc_type.lower() == 'gcn':
                layers.append((nn.Dropout(drop_rate), 'x -> x'), )
                layers.append((GCNConv(in_dim, out_dim), 'x, edge_index, edge_weight -> x'),)
                layers.append((BatchNorm(out_dim, momentum=batchnorm_mm), 'x -> x'))
            elif enc_type.lower() == 'gat':
                layers.append((nn.Dropout(drop_rate), 'x -> x'), )
                layers.append((GATConv(in_dim, out_dim), 'x, edge_index, edge_weight -> x'),)
                layers.append((BatchNorm(out_dim, momentum=batchnorm_mm), 'x -> x'))
            elif enc_type.lower() == 'sage':
                layers.append((nn.Dropout(drop_rate), 'x -> x'), )
                layers.append((GraphConv(in_dim, out_dim, aggr='mean'), 'x, edge_index, edge_weight -> x'),)
                layers.append((BatchNorm(out_dim, momentum=batchnorm_mm), 'x -> x'))
            elif enc_type.lower() == 'gine':
                layers.append((nn.Dropout(drop_rate), 'x -> x'), )
                layers.append((GINEConv(Sequential(
                    Linear(in_dim, out_dim), nn.ReLU(),
                    Linear(out_dim, out_dim), nn.ReLU(),
                    BatchNorm()
                )), 'x, edge_index, edge_weight -> x'))

            if act_fn is None:
                pass
            else:
                layers.append(act_fn)

        if enc_type.lower() == 'linear':
            self.model = nn.Sequential(*layers)
        else:
            self.model = Sequential('x, edge_index, edge_weight', layers)

        self.reset_parameters()
    
    def reset_parameters(self):
        try:
            self.model.reset_parameters()
        except:
            # kaiming_uniform
            for m in self.modules():
                if isinstance(m, Linear):
                    m.reset_parameters()

    def forward(self, inputs):
        if self.enc_type.lower() == 'linear':
            out = self.model(inputs)
        else:
            x = inputs['x']
            edge_index = inputs['edge_index']
            edge_weight = inputs['edge_attr']
            out = self.model(x, edge_index, edge_weight)
        return out


class AutoEncoder_Decoder(nn.Module):
    def __init__(self, input_dim, layer_sizes,
                 act_fn=nn.ReLU(), drop_rate=0.0, batchnorm_mm=0.99):
        """AutoEncoder Decoder Block"""
        super(AutoEncoder_Decoder, self).__init__()

        hid_dims = layer_sizes[::-1]

        layers = []
        for in_dim, out_dim in zip(hid_dims[:-1], hid_dims[1:]):
            layers.append(nn.Dropout(drop_rate))
            layers.append(Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim, momentum=batchnorm_mm))
            if act_fn is None:
                pass
            else:
                layers.append(act_fn)
        layers.append(Linear(out_dim, input_dim))
        self.model = nn.Sequential(*layers)

        self.reset_parameters()
    
    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, Linear):
                m.reset_parameters()

    def forward(self, x):
        out = self.model(x)
        return out
    


class AE(nn.Module):
    def __init__(self, input_dim, layer_sizes, enc_type, dropout):
        super(AE, self).__init__()
        self.layer_sizes = eval(layer_sizes)

        enc_layer_sizes = copy(self.layer_sizes)
        dec_layer_sizes = copy(self.layer_sizes)

        self.encoder = AutoEncoder_Encoder(
            input_dim, enc_layer_sizes, enc_type,
            drop_rate=dropout
        )

        self.decoder = AutoEncoder_Decoder(
            input_dim, dec_layer_sizes,
            drop_rate=dropout
        )

    def encode(self, x):
        # posterior = self.encoder(x)
        # return posterior
        z = self.encoder(x)
        return z

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def forward(self, x):
        # posterior = self.encode(x)
        # z = posterior.mode()
        z = self.encode(x)
        recon = self.decode(z)
        # return recon, posterior
        return recon, z


class GAE(nn.Module):
    def __init__(self, input_dim, layer_sizes, enc_type, dropout):
        super(GAE, self).__init__()
        self.layer_sizes = eval(layer_sizes)

        enc_layer_sizes = copy(self.layer_sizes)
        dec_layer_sizes = copy(self.layer_sizes)

        self.encoder = AutoEncoder_Encoder(
            input_dim, enc_layer_sizes, enc_type,
            drop_rate=dropout
        )

        self.decoder = AutoEncoder_Decoder(
            input_dim, dec_layer_sizes,
            drop_rate=dropout
        )

    def encode(self, g):
        # posterior = self.encoder(g)
        # return posterior
        z = self.encoder(g)
        return z

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def forward(self, g):
        # posterior = self.encode(g)
        # z = posterior.mode()
        z = self.encode(g)
        recon = self.decode(z)
        # return recon, posterior
        return recon, z
