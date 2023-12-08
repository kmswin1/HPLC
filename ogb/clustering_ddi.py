import networkx as nx
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
import random
import numpy as np
from fluidc import asyn_fluidc
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

k = int(sys.argv[1])
d_names = ['ogbl-ppa', 'ogbl-collab', 'ogbl-ddi', 'ogbl-citation2']
d_name = d_names[2]
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
device = 'cpu'
num_local = 8
dataset = PygLinkPropPredDataset(name=d_name,
                                 transform=T.ToSparseTensor())
data = dataset[0]
adj_t = data.adj_t.to(device)

data = data.to(device)

adj = adj_t.to_scipy()
net = nx.from_scipy_sparse_matrix(adj)
se = []
row, col, _ = adj_t.coo()
edge_index = torch.stack([col, row], dim=0)

nets = sorted(nx.connected_components(net), key=len, reverse=True)
if len(nets) > 1:
    lgst_net = nx.subgraph(net, nets[0])
    com = asyn_fluidc(lgst_net, k=num_local)
else:
    com = asyn_fluidc(net, k=num_local)
anchor_nodes = []
node2cluster = {}

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
torch.save(lap_pe, 'ddi_lap_pe_async_fluid.pt')
torch.save(anchor_nodes, 'ddi_anchor_async_fluid.pt')
pos_enc = []
for node in range(data.num_nodes):
    temp = []
    if node % 1000 == 0:
        print (f"{node} node completed!!")
    for anchor_node in anchor_nodes:
        try:
            temp.append(len(nx.algorithms.shortest_path(net, node, anchor_node)))
        except:
            temp.append(0)
    pos_enc.append(np.array(temp))
pos_enc = np.array(pos_enc)
max_d = 0
for p in pos_enc:
    max_d = max(max_d, max(p))

np.where(pos_enc == 0, max_d, pos_enc)
for i, a in enumerate(anchor_nodes):
    pos_enc[a][i] = 0

if len(nets) > 1:
    lgst_net = nx.subgraph(net, nets[0])
    com = asyn_fluidc(lgst_net, k=num_local)
else:
    com = asyn_fluidc(net, k=num_local)

torch.save(com, 'ddi_com_async_fluid.pt')
torch.save(pos_enc, 'ddi_pe_async_fluid.pt')
