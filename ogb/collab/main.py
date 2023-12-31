import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import torch.nn as nn
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

class HPLC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(HPLC, self).__init__()

        self.args = args
        self.pos_embedding = nn.Linear(2*args.dim-1, args.num_local)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.fc1 = torch.nn.ModuleList()
        for _ in range(self.args.num_local):
            self.fc1.append(nn.Linear(in_channels, hidden_channels))
        self.fc2 = torch.nn.ModuleList()
        for _ in range(self.args.num_local):
            self.fc2.append(nn.Linear(hidden_channels, hidden_channels))
        self.fc3 = torch.nn.ModuleList()
        for _ in range(self.args.num_local):
            self.fc3.append(nn.Linear(hidden_channels, in_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.fc1:
            lin.reset_parameters()
        for lin in self.fc2:
            lin.reset_parameters()
        for lin in self.fc3:
            lin.reset_parameters()

    def forward(self, x, adj_t, com_xs, pos_emb, lap_pe):
        pos_emb = self.pos_embedding(torch.cat([pos_emb, lap_pe], dim=-1))
        x = torch.cat([x, pos_emb], dim=-1)
        for i, com_x in enumerate(com_xs):
            x_temp = x[com_x]
            x_temp = self.fc1[i](x_temp)
            x_temp = F.leaky_relu(x_temp)
            x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
            x_temp = self.fc2[i](x_temp)
            x_temp = F.leaky_relu(x_temp)
            x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
            x_temp = self.fc3[i](x_temp)
            x_temp = F.leaky_relu(x_temp)
            x_temp = F.dropout(x_temp, p=self.dropout, training=self.training)
            x[com_x] = x_temp

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.struct_embedding = nn.Sequential(nn.Linear(32,32),
                                              nn.ReLU(),
                                              nn.Linear(32,1)
                                              )

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, se_i, se_j):
        x_i = torch.cat([x_i, se_i], dim=-1)
        x_j = torch.cat([x_j, se_j], dim=-1)
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        sign_flip = torch.rand(data.lap_pe.size(1)).to(data.x.device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        flipped_lap_pe = data.lap_pe*sign_flip.unsqueeze(0)

        h = model(data.x, data.adj_t, data.com_x, data.pos_enc, flipped_lap_pe)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)

        neg_out = predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t, data.com_x, data.pos_enc, data.lap_pe)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(data.x, data.adj_t, data.com_x, data.pos_enc, data.lap_pe)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]], data.se[edge[0]], data.se[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [50]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64*1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--num_local', type=int, default=12)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-collab')
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)

    split_edge = dataset.get_edge_split()

    full_edge_index = edge_index
    data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
    data.full_adj_t = data.full_adj_t.to_symmetric()


    pe = torch.load('collab_pe_async_fluid.pt')
    com = torch.load('collab_com_async_fluid.pt')
    se = torch.load('collab_se.pt')

    if type(com) != list:
        com = list(com.communities)

    num_local = len(com)
    args.num_local = num_local
    args.dim = pe.shape[1]

    data['com_x'] = []
    for j, c in enumerate(com):
        data['com_x'].append(list(c))

    data['pos_enc'] = torch.tensor(pe, dtype=torch.float)
    data['se'] = torch.tensor(se, dtype=torch.float)
    lap_pe = torch.load('collab_lap_pe_async_fluid.pt')

    data['lap_pe'] = torch.tensor(lap_pe, dtype=torch.float)

    data = data.to(device)

    model = HPLC(data.num_features + args.num_local, args.hidden_channels,
                args.hidden_channels, args.num_layers,
                args.dropout, args).to(device)

    predictor = LinkPredictor(args.hidden_channels + se.shape[1], args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@50': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):

            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        loggers[key].write_statistics()


if __name__ == "__main__":
    main()
