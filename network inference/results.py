import cvxpy as cvx 
import numpy as np

# read in the save intermediate data
adj = np.load('data/adj.npy')
alpha = np.load('alpha.npy')
accuracy = np.mean(np.abs(alpha[adj != 0] - adj[adj != 0]))
min_tol = 1e-4
recall = np.sum(np.logical_and(alpha >min_tol, adj >min_tol))/np.sum(adj >min_tol)
precision = np.sum(adj > min_tol)/np.sum(alpha >min_tol)
print(accuracy)
print(recall)
print(precision)
