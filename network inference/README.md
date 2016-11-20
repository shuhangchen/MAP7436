This repo contains my implementation of Dr. Manuel Gomez Rodriguez`s paper, 'Uncovering the Temporal Dynamics of Diffusion Networks'.

1. During implementation, I referred the original matlab code Netrate in the author's webpage.
2. I meant to implement the algorithm with Torch, but then I realized it might be a little complicated to optimize the log term with gradient decent. We could use coordinate decent if we must. At last, I turned to the CVXPY package, which is the python version of the CVX package the author used.
3. As a result, the code is an ugly mixture of torch and python, but it is much faster than Matlab codes. I tested it on the simulation data provided by the author in Netrate package.
