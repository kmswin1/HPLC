import networkx as nx
import torch
import numpy as np
from fluidc import asyn_fluidc
import sys
from scipy import sparse as sp
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils.convert import from_networkx

data = str(sys.argv[1])
k=int(sys.argv[2])
if data == 'facebook':
    num_local = 8

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

np.random.seed(123)
device = 'cpu'
filename = f'data/facebook.txt'
g = nx.read_edgelist(filename,create_using=nx.Graph(), nodetype = int,  data=(("weight", float),))
adj_label = nx.adjacency_matrix(g, nodelist = sorted(g.nodes())) 
adj = (adj_label > 0).astype('int') # to binary
net = nx.from_scipy_sparse_matrix(adj)

adj_train_coo = adj.tocoo()
edge_index = np.concatenate((adj_train_coo.row[np.newaxis,:],adj_train_coo.col[np.newaxis,:]), axis=0)
edge_index = torch.LongTensor(edge_index)

nets = sorted(nx.connected_components(net), key=len, reverse=True)
if len(nets) > 1:
    lgst_net = nx.subgraph(net, nets[0])
    com = asyn_fluidc(lgst_net, k=k*num_local)
else:
    com = asyn_fluidc(net, k=k*num_local)
anchor_nodes = []
torch.save(com, 'data/'+data+'_com_async_fluid.pt')
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
lap_pe = np.concatenate([lap_pe, np.random.rand(k*num_local-1).reshape(1,-1)], axis=0)
lap_pe = lap_pe[membership]
torch.save(lap_pe, 'data/'+data+'_lap_pe_async_fluid.pt')
torch.save(anchor_nodes, 'data/'+data+'_anchor_async_fluid.pt')
pos_enc = []
for node in range(len(net.nodes())):
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

torch.save(com, 'data/'+data+'_com_async_fluid.pt')
torch.save(pos_enc, 'data/'+data+'_pe_async_fluid.pt')