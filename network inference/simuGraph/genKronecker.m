% this script generates the Kronecker graph
% shuhang
clear all
clc
K1_random = [0.5 0.5 ; 0.5 0.5]; % renyi
K1_core = [0.9 0.5 ; 0.5 0.1];
K1_hier = [0.9 0.1; 0.9 0.1];

K_random = K1_random;
K_core= K1_core;
K_hier = K1_hier;


for i=1:5  % size of the adjancy matrix is 
    K_random = kron(K_random, K1_random);
    K_core = kron(K_core, K1_core);
    K_hier = kron(K_hier, K1_hier);
end

% then we sample the edges
K_r = binornd(1, K_random);
K_c = binornd(1, K_core);
K_h = binornd(1, K_hier);

% then we sample the adj matrices
% random matrix
adj_Kr = zeros(64, 64);
index_Kr = find(K_r ~= 0);
adj_Kr(index_Kr) = 0.01 + (1-0.01).* rand(1,size(index_Kr,1));
% core
adj_Kc = zeros(64, 64);
index_Kc = find(K_c ~= 0);
adj_Kc(index_Kc) = 0.01 + (1-0.01).* rand(1,size(index_Kc,1));
% hire
adj_Kh = zeros(64, 64);
index_Kh = find(K_h ~= 0);
adj_Kh(index_Kh) = 0.01 + (1-0.01).* rand(1,size(index_Kh,1));

save Kc_adj.mat adj_Kc 
save Kh_adj.mat adj_Kh
save Kr_adj.mat adj_Kr

