import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

class HPLC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(HPLC, self).__init__()

        self.args = args
        #self.emb = nn.Embedding(2*self.args.num_local, 5)
        self.pos_embedding = nn.Sequential(nn.Linear(2*args.dim-1, 2*args.dim-1),
                                           nn.ReLU(),
                                            nn.Linear(2*args.dim-1, args.num_local)
                                            )
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

def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size, data, pos_emb, lap_pe, se):
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        sign_flip = torch.rand(lap_pe.size(1)).to(x.device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        flipped_lap_pe = lap_pe*sign_flip.unsqueeze(0)
        h = model(x, adj_t, data.com_x, pos_emb, flipped_lap_pe, se)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size, data, pos_emb, lap_pe, se):
    model.eval()
    predictor.eval()

    h = model(x, adj_t, data.com_x, pos_emb, lap_pe, se)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]], se[edge[0]], se[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20]:
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
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--num_local', type=int, default=8)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    pe = torch.load('ddi_pe_async_fluid.pt')
    com = torch.load('ddi_com_async_fluid.pt')
    lap_pe = torch.load('ddi_lap_pe_async_fluid.pt')
    se = torch.load('ddi_se.pt')

    data['lap_pe'] = torch.tensor(lap_pe, dtype=torch.float).to(args.device)

    if type(com) != list:
        com = list(com.communities)

    num_local = len(com)
    args.num_local = num_local
    args.dim = pe.shape[1]
    data['com_x'] = []
    for j, c in enumerate(com):
        data['com_x'].append(list(c))

    data['pos_enc'] = torch.tensor(pe, dtype=torch.float).to(device)
    data['se'] = torch.tensor(se, dtype=torch.float).to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}


    model = HPLC(args.hidden_channels + args.num_local, args.hidden_channels,
                args.hidden_channels, args.num_layers,
                args.dropout, args).to(device)

    emb = nn.Embedding(data.adj_t.size(0),
                           args.hidden_channels).to(device)


    predictor = LinkPredictor(args.hidden_channels + se.shape[1], args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@20': Logger(args.runs, args),
    }

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):

            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, args.batch_size, data, data.pos_enc, data.lap_pe, data.se)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, args.batch_size, data, data.pos_enc, data.lap_pe, data.se)
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

            h = model(emb.weight, adj_t, data.com_x, data.pos_enc, data.lap_pe, data.se)
            torch.save(h.detach().cpu().numpy(), 'embedding_output.pt')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        loggers[key].write_statistics()


if __name__ == "__main__":
    main()