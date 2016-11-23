This repo contains my implementation of Dr. Manuel Gomez Rodriguez`s paper, 'Uncovering the Temporal Dynamics of Diffusion Networks'.

## Implementation
1. During implementation, I referred the original matlab code Netrate in the author's webpage.
2. I meant to implement the algorithm with Torch, but then I realized it might be a little complicated to optimize the log term with gradient decent. We could use coordinate decent if we must. At last, I turned to the CVXPY package, which is the python version of the CVX package the author used.
3. As a result, the code is an ugly mixture of torch and python, but it is much faster than Matlab codes. I tested it on the simulation data provided by the author in Netrate package.

## Simulation
I also simulated the activation propagation process in simulateCas.lua.
1. There are two types of networks in the simulation, Kronecker network provided by Dr. Gomez in his paper, Renyi generated by the CONTEST package whose transmission rates are uniformly distributed in (0,1).

## Results
Data description:
  - All graph have 1024 nodes.
  - Kronecker has 4084 edges, Renyi_1 has 4096 edges, and Renyi_1-5 has 6144 edges.
  - Nan means I have not gotten the results yet.


**Table:** Each column means the recovery accuracy of one type of network under different amount of cascades. The first number in the tuple (a,b) is mean absolute difference \(a = \sum_{i,j}|\alpha_{ij} - \hat \alpha_{ij}|\), the second element is relative absolute difference \(b = \frac{\sum_{i,j}|\alpha_{ij} - \hat \alpha_{ij}|}{\alpha_{ij}}\).

Network Cascades | Renyi_1 | Kronecker | Renyi_1-5
--- | --- | --- |---
1000 |(0.4, 1.91) | (0.087, 0.756) | Nan
5000 | (0.36, 1.99) | (0.038,0.331)| Nan
10000 | Nan | (0.026, 0.24) | Nan