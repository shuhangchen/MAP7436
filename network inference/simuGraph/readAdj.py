# This script reads adjancy matr
try:
    import matplotlib.pyplot as plt
except:
    raise
import networkx as nx
import numpy as np
import scipy.io as sio

adj = sio.loadmat('adj_1.5_64.mat')['adj']
size = 64
# add each node and edge into networkx
DG = nx.DiGraph()
# add all nodes
DG.add_nodes_from(range(0,size))
# add all edges
it = np.nditer(adj, flags=['multi_index'])
while not it.finished:
    index = (it.multi_index[0], it.multi_index[1])
    if adj[index[0]][index[1]] > 0:
        DG.add_edge(index[0], index[1], weight = adj[index[0]][index[1]])
    it.iternext()
# draw the graph

nx.draw(DG, arrows=False)
plt.savefig('graphRenyi15.png')
plt.axis('off')
plt.show()
