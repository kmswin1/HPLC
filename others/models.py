import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge, GATConv

class HPLC(nn.Module):
    def __init__(self, dim_feat, dim_h, dim_z, dropout, num_local, dim_pos, gnn_type='GCN', jk_mode='mean', dec='mlp'):
        super(HPLC, self).__init__()
        gcn_num_layers = 3
        self.encoder = GNN(dim_feat, dim_h, dim_z, dropout, num_local, dim_pos, gnn_type=gnn_type, num_layers = gcn_num_layers, jk_mode=jk_mode)
        if jk_mode == 'cat':
            dim_in = dim_h * (gcn_num_layers-1) + dim_z
        else:
            dim_in = dim_z
        self.decoder = Decoder(dec, dim_in)
        self.init_params()

    def forward(self, adj, features, edges, com_xs, pos_emb, lap_pe):
        z = self.encoder(adj, features, com_xs, pos_emb, lap_pe)
        z_i = z[edges.T[0]]
        z_j = z[edges.T[1]]
        res = self.decoder(z_i, z_j)
        return res

    def init_params(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()


class GNN(nn.Module):
    def __init__(self, dim_feat, dim_h, dim_z, dropout, num_local, dim_pos, gnn_type='GCN', num_layers=3, jk_mode='mean', batchnorm=True):
        super(GNN, self).__init__()
        self.pos_embedding = nn.Linear(2*dim_pos-1, num_local)
        assert jk_mode in ['max','sum','mean','lstm','cat','none']
        self.act = nn.ELU()
        self.dropout = dropout
        self.linear = torch.nn.Linear(dim_h, dim_z)

        dim_feat += num_local

        if gnn_type == 'SAGE':
            gnnlayer = SAGEConv
        elif gnn_type == 'GCN':
            gnnlayer = GCNConv
        elif gnn_type == 'GAT':
            gnnlayer = GATConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(gnnlayer(dim_feat, dim_h))
        for _ in range(num_layers - 2):
            self.convs.append(gnnlayer(dim_h, dim_h))
        self.convs.append(gnnlayer(dim_h, dim_z))
        self.projs = torch.nn.ModuleList()
        for _ in range(num_local):
            self.projs.append(nn.Linear(dim_feat, dim_feat))
        self.fc1 = torch.nn.ModuleList()
        for _ in range(num_local):
            self.fc1.append(nn.Linear(dim_feat, dim_feat))
        self.fc2 = torch.nn.ModuleList()
        for _ in range(num_local):
            self.fc2.append(nn.Linear(dim_feat, dim_feat))
        self.fc3 = torch.nn.ModuleList()
        for _ in range(num_local):
            self.fc3.append(nn.Linear(dim_feat, dim_feat))

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim_h) for _ in range(num_layers)])

        self.jk_mode = jk_mode
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=dim_h, num_layers=num_layers)
        elif self.jk_mode == 'mean':
            self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def forward(self, adj, features, com_xs, pos_emb, lap_pe):
        pos_emb = self.pos_embedding(torch.cat([pos_emb, lap_pe], dim=-1))
        features = torch.cat([features, pos_emb], dim=-1)
        for i, com_x in enumerate(com_xs):
            x_temp = features[com_x]
            x_temp = self.fc1[i](x_temp)
            x_temp = F.leaky_relu(x_temp)
            x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
            x_temp = self.fc2[i](x_temp)
            x_temp = F.leaky_relu(x_temp)
            x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
            x_temp = self.fc3[i](x_temp)
            x_temp = F.leaky_relu(x_temp)
            x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
            features[com_x] = x_temp
        out = features

        for conv in self.convs[:-1]:
            out = conv(out, adj)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.convs[-1](out, adj)

        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk.reset_parameters()
        for lin in self.fc1:
            torch.nn.init.xavier_normal_(lin.weight)
        for lin in self.fc2:
            torch.nn.init.xavier_normal_(lin.weight)
        for lin in self.fc3:
            torch.nn.init.xavier_normal_(lin.weight)


class Decoder(nn.Module):
    def __init__(self, dec, dim_z=256, dim_h=256):
        super(Decoder, self).__init__()
        self.dec = dec
        if dec == 'innerproduct':
            dim_in = 1
        elif dec == 'mlp':
            dim_in = dim_z
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, dim_h, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim_h, 1, bias=True)
        )

    def forward(self, z_i, z_j):
        z = z_i * z_j
        h = self.mlp_out(z).squeeze()
        return torch.sigmoid(h)

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue