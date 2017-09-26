import cvxpy as cvx 
import numpy as np

# read in the save intermediate data
adj = np.load('npy_data/adj.npy')
cascade = np.load('npy_data/cascade.npy')
phi_1 = np.load('npy_data/phi_1.npy')
phi_2 = np.load('npy_data/phi_2.npy')
numNodes = int(np.load('npy_data/numNodes.npy')[0])
numCascades = np.load('npy_data/numCascades.npy').astype(int)

phi_3 = np.zeros(phi_1.shape)
alpha = np.zeros((numNodes, numNodes))
print('Optimizing nodes')
for i in range(numNodes):
    # print('dealing with the {0}th node ... \n'.format(i))
    if (numCascades[i] == 0):
        phi_3[:, i] = 0
        continue
    
    alpha_i = cvx.Variable(numNodes)
    obj = 0

    for j in range(numNodes):
        if (phi_2[j][i] > 0):
            obj +=  - alpha_i[j] * ( phi_2[j][i] + phi_1[j][i])
    for c in range(cascade.shape[0]):
        idx = np.nonzero(cascade[c][:] != -1)[0]
        val = np.sort(cascade[c][idx])
        ord = np.argsort(cascade[c][idx])
        idx_ord = idx[ord]
        cidx = np.nonzero(idx_ord == i)[0]
        if (cidx.size > 0 and cidx[0] > 0):
            obj += cvx.log(cvx.sum_entries(alpha_i[idx_ord[0:cidx[0]]]))
    constraints = []
    for j in range(numNodes):
        if phi_2[j][i] == 0 :
            constraints.append(alpha_i[j] == 0)
        else:
            constraints.append(alpha_i[j] >= 0)
    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver = cvx.SCS)
    alpha[:,i] = alpha_i.value.flatten()

# compute accuracy
adj = np.transpose(adj)
accuracyRel = np.mean(np.abs(alpha[adj != 0] - adj[adj != 0])/adj[adj != 0])
accuracy = np.mean(np.abs(alpha[adj != 0] - adj[adj != 0]))
min_tol = 1e-4
recall = float(np.sum(np.logical_and(alpha >min_tol, adj >min_tol)))/np.sum(adj >min_tol)
precision = float(np.sum(adj > min_tol))/np.sum(alpha >min_tol)
print('The accuracy is {}'.format(accuracy))
print('The relative accuracy is {}'.format(accuracyRel))
print('Recall is {}'.format(recall))
print('Precsion is {}'.format(precision))
np.save('npy_data/alpha.npy', alpha)


    
