import networkx as nx
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
import random
import numpy as np
from fluidc import asyn_fluidc
import multiprocessing
import sys
from scipy import sparse as sp
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils.convert import from_networkx
def positional_encoding(A, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    D = sp.diags(degree(A[0]).numpy() ** -0.5, dtype=float)
    A = to_dense_adj(A).squeeze(0).numpy()
    L = sp.eye(pos_enc_dim) - D * A * D

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pe = torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float() 

    return lap_pe
num_local=15
k = int(sys.argv[1])

def get_com():
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    adj = data.adj_t.to_symmetric().to_scipy()
    net = nx.from_scipy_sparse_matrix(adj)
    edge_index = adj
    row, col = edge_index.row, edge_index.col
    edge_index = torch.stack([torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long)])

    nets = sorted(nx.connected_components(net), key=len, reverse=True)
    if len(nets) > 1:
        lgst_net = nx.subgraph(net, nets[0])
        others = []
        for n in nets[1:]:
            others.extend(list(n))
        com = asyn_fluidc(lgst_net, k=num_local)
    else:
        com = asyn_fluidc(net, k=num_local)
    node2cluster = {}
    anchor_nodes = []

    for i, c in enumerate(com):
        subnet = nx.subgraph(net, c)
        local_com = asyn_fluidc(subnet, k)
        for j, local_c in enumerate(local_com):
            local_subnet = nx.subgraph(subnet, local_c)
            anchor_node = sorted(local_subnet.degree(), key=lambda x : x[1], reverse=True)[0][0]
            anchor_nodes.append(anchor_node)
            for node in local_c:
                node2cluster[node] = i

    membership = []

    for i in range(len(net)):
        try:
            membership.append(node2cluster[i])
        except:
            membership.append(len(anchor_nodes))

    anchor_distances = {}
    for i in range(len(anchor_nodes)):
        for j in range(i+1, len(anchor_nodes)):
            dist = len(nx.algorithms.shortest_path(net, anchor_nodes[i], anchor_nodes[j]))
            anchor_distances[(i,j)] = dist
            anchor_distances[(j,i)] = dist
            
    landmark_graph = nx.Graph()
    temperature=5.0
    for i in range(len(anchor_nodes)):
        for j in range(i+1, len(anchor_nodes)):
            dist = anchor_distances[(i,j)]
            weight = np.exp((-dist**2)/temperature)
            landmark_graph.add_edge(i, j, weight=weight)
            landmark_graph.add_edge(j, i, weight=weight)

    landmark_graph = from_networkx(landmark_graph)

    lap_pe = positional_encoding(landmark_graph.edge_index, k*num_local)
    lap_pe = np.concatenate([lap_pe, np.zeros(k*num_local-1).reshape(1,-1)], axis=0)
    lap_pe = lap_pe[membership]
    torch.save(lap_pe, 'citation2_lap_pe_async_fluid.pt')
    
    if len(nets) > 1:
        lgst_net = nx.subgraph(net, nets[0])
        com = asyn_fluidc(lgst_net, k=num_local)
    else:
        com = asyn_fluidc(net, k=num_local)
    torch.save(com, 'citation2_com_async_fluid.pt')
    torch.save(anchor_nodes, 'citation2_anchor_async_fluid.pt')

    return [i for i in range(data.num_nodes)], net, anchor_nodes, edge_index

def get_pe(net, anchor_nodes, nodes, map_ret):
    for node in nodes:
        pos_ret = dict()
        pos_enc = []
        for anchor_node in anchor_nodes:
            try:
                pos_enc.append(len(nx.algorithms.shortest_path(net, node, anchor_node)))
            except:
                pos_enc.append(0)
        pos_ret[node] = pos_enc
        map_ret.update(pos_ret)
        print(f"{node} node completed!!")

def main():
    mult_manager = multiprocessing.Manager()
    nodes, net, anchor_nodes, edge_index = get_com()
    WORKER_NUM = 128
    bs = int(len(nodes) / WORKER_NUM)
    if len(nodes) % WORKER_NUM != 0:
        bs += 1
    vec_process = []
    return_dict = mult_manager.dict()
    for pidx in range(WORKER_NUM):
        p = multiprocessing.Process(target=get_pe, args=(net, anchor_nodes, nodes[pidx * bs: min((pidx + 1) * bs, len(nodes))], return_dict))
        p.start()
        vec_process.append(p)
    for p in vec_process:
        p.join()

    ret = dict()
    ret.update(return_dict.copy())
    torch.save(ret, 'citation2_pe_dict.pt')
    pos_enc = []
    for i in range(len(nodes)):
        pos_enc.append(ret[i])
    pos_enc = np.array(pos_enc)
    max_d = 0
    for p in pos_enc:
        max_d = max(max_d, max(p))

    np.where(pos_enc == 0, max_d, pos_enc)
    for i, a in enumerate(anchor_nodes):
        pos_enc[a][i] = 0

    new_pos_enc = []
    for i, p in enumerate(pos_enc):
        temp_p = list(sorted(p))
        new_pos_enc.append(np.array(temp_p[:num_local]))

    new_pos_enc = np.array(new_pos_enc)
    torch.save(new_pos_enc, 'citation2_pe_async_fluid.pt')

if __name__ == '__main__':
    main()
