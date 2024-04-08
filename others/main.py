import os
import sys
import time
import argparse
import torch
import torch_sparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import networkx as nx
import numpy as np
from torch_geometric.data import Data

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *

def get_args():
    parser = argparse.ArgumentParser(description='HPLC')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--embraw', type=str, default='none')
    parser.add_argument('--neg_rate', type=int, default=1, help='rate of negative samples during training')
    parser.add_argument('--dec', type=str, default='mlp', choices=['innerproduct','hadamard','mlp'], help='choice of decoder')
    parser.add_argument('--seed', type=int, default=-1, help='fix random seed if needed')
    parser.add_argument('--verbose', type=int, default=1, help='whether to print per-epoch logs')
    parser.add_argument('--trails', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-1, help='-2 for CPU, -1 for default GPU, >=0 for specific GPU')
    parser.add_argument('--n_workers', type=int, default=20, help='number of CPU processes for finding counterfactual links in the first run')
    parser.add_argument('--gnn_type', type=str, default='GCN',choices=['SAGE','GCN', 'GAT'])
    parser.add_argument('--jk_mode', type=str, default='mean',choices=['max','cat','mean','lstm','sum','none'])
    parser.add_argument('--dim_h', type=int, default=256)
    parser.add_argument('--dim_z', type=int, default=256)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1024*64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_ft', type=float, default=5e-3)
    parser.add_argument('--l2reg', type=float, default=5e-6)
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of edges for validation set (and same number of no-edges)')
    parser.add_argument('--test_frac', type=float, default=0.2, help='fraction of edges for testing set (and same number of no-edges)')
    parser.add_argument('--metric', type=str, default='auc', help='main evaluation metric')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:0' if args.gpu >= -1 else 'cpu')

    return args


def train(args, logger):
    pe = torch.load('data/' + args.dataset + '_pe_async_fluid.pt')
    com = torch.load('data/' + args.dataset + '_com_async_fluid.pt')
    lap_pe = torch.load('data/' + args.dataset + '_lap_pe_async_fluid.pt')
    num_local = len(com)
    data = Data()
    data['com_x'] = []
    for j, c in enumerate(com):
        data['com_x'].append(list(c))

    data['lap_pe'] = torch.tensor(lap_pe, dtype=torch.float).to(args.device)
    data['pos_enc'] = torch.tensor(pe, dtype=torch.float).to(args.device)
    dim_pos = pe.shape[1]

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # load data
    adj_label, features, dim_feat, adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false = load_data(args, logger)
    # load n by n treatment matrix

    # get train_edges and train_edges_false
    logger.info('...getting train splitting...')
    trainsplit_dir_name = 'data/train_split/'
    if not os.path.exists(trainsplit_dir_name):
        os.makedirs(trainsplit_dir_name, exist_ok=True)
    try:
        train_edges, train_edges_false = pickle.load(open(f'{trainsplit_dir_name}{args.dataset}.pkl', 'rb'))
    except:
        train_edges = np.asarray(sp.triu(adj_train, 1).nonzero()).T
        all_set = set([tuple(x) for x in train_pairs])
        edge_set = set([tuple(x) for x in train_edges])
        noedge_set = all_set - edge_set
        train_edges_false = np.asarray(list(noedge_set))
        pickle.dump((train_edges, train_edges_false), open(f'{trainsplit_dir_name}{args.dataset}.pkl', 'wb'))

    assert train_edges.shape[0] + train_edges_false.shape[0] == train_pairs.shape[0]
    logger.info(f'train_edges len: {train_edges.shape[0]}, batch size: {args.batch_size}')
    logger.info('...finishing train splitting...')

    max_neg_rate = train_edges_false.shape[0] // train_edges.shape[0] - 1
    if args.neg_rate > max_neg_rate:
        args.neg_rate = max_neg_rate
        logger.info(f'negative rate change to: {max_neg_rate}')
    val_pairs = np.concatenate((val_edges, val_edges_false), axis=0)
    val_labels = np.concatenate((np.ones(val_edges.shape[0]), np.zeros(val_edges_false.shape[0])), axis=0)
    test_pairs = np.concatenate((test_edges, test_edges_false), axis=0)
    test_labels = np.concatenate((np.ones(test_edges.shape[0]), np.zeros(test_edges_false.shape[0])), axis=0)

    # cast everything to proper type
    adj_train_coo = adj_train.tocoo()
    edge_index = np.concatenate((adj_train_coo.row[np.newaxis,:],adj_train_coo.col[np.newaxis,:]), axis=0)
    adj_norm = torch_sparse.SparseTensor.from_edge_index(torch.LongTensor(edge_index))

    # move everything to device
    device = args.device
    adj_norm = adj_norm.to(device)
    features = features.to(device)

    model = HPLC(dim_feat, args.dim_h, args.dim_z, args.dropout, num_local, dim_pos, args.gnn_type, args.jk_mode, args.dec)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.l2reg)

    logger.info(f'Using evaluation metric: {args.metric}')
    best_val_res = 0.0
    cnt_wait = 0
    test_res = {}
    for epoch in range(args.epochs):
        total_examples = 0
        total_loss = 0
        for perm in DataLoader(range(train_edges.shape[0]), args.batch_size, shuffle=True):
            # sample no_edges for this batch
            pos_edges =  train_edges[perm]
            neg_sample_idx = np.random.choice(train_edges_false.shape[0], args.neg_rate * len(perm), replace=False)
            neg_edges = train_edges_false[neg_sample_idx]
            # move things to device
            model.train()
            optim.zero_grad()
            # forward pass
            pos_score = model(adj_norm, features, pos_edges, data.com_x, data.pos_enc, data.lap_pe)
            neg_score = model(adj_norm, features, neg_edges, data.com_x, data.pos_enc, data.lap_pe)

            # loss
            loss = -torch.log(pos_score + 1e-15).mean() + -torch.log(1- neg_score + 1e-15).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item() * pos_edges.shape[0]
            total_examples += pos_edges.shape[0]

        total_loss /= total_examples
        #evaluation
        model.eval()
        with torch.no_grad():
            z = model.encoder(adj_norm, features, data.com_x, data.pos_enc, data.lap_pe)
            logits_val = model.decoder(z[val_pairs.T[0]], z[val_pairs.T[1]]).detach().cpu()
            logits_test = model.decoder(z[test_pairs.T[0]], z[test_pairs.T[1]]).detach().cpu()
        val_res = eval_ep_batched(logits_val, val_labels, val_edges.shape[0])
        if val_res[args.metric] >= best_val_res:
            cnt_wait = 0
            best_val_res = val_res[args.metric]
            test_res = eval_ep_batched(logits_test, test_labels, test_edges.shape[0])
            test_res['best_val'] = val_res[args.metric]
            if args.verbose:
                logger.info('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f} test: {:.4f}'.format(
                    epoch+1, total_loss, args.lr, val_res[args.metric], test_res[args.metric]))
        else:
            cnt_wait += 1
            if args.verbose:
                logger.info('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f}'.format(
                    epoch+1, total_loss, args.lr, val_res[args.metric]))

        if cnt_wait >= args.patience:
            if args.verbose:
                print('Early stopping!')
            break

    return test_res

def main(args):
    all_results = []
    #for run in range(10):
    args.seed = args.seed + 1
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{time.strftime("%m-%d_%H-%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)

    logger.info(f'Input argument vector: {args.argv[1:]}')
    logger.info(f'args: {args}')
    results = {'auc': [], 'ap': [], 'best_val': []}
    for _ in range(args.trails):
        res = train(args, logger)
        for metric in results.keys():
            results[metric].append(res[metric])
    logger.info('final results:')
    for metric, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            metric, np.mean(nums), np.std(nums), nums))
        with open('results.txt', 'a') as f:
            f.write(f"{args.dataset}, {metric}, {np.mean(nums)}, {np.std(nums)}\n")

    #logger.info(f'all run final results: {np.mean(all_results)} +- {np.std(all_results)}')


if __name__ == "__main__":
    args = get_args()
    main(args)
