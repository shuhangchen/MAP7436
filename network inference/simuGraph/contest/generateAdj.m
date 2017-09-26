% this script generates an adjancy matrix using erdrey function
% and for each nonzero edge, we draw a uniform distribution as their
% transmission rates
% shuhang
clc
clear
numNodes = 64;
coef = [1 ];
for i=1:1
    net = erdrey(numNodes, coef(i)*numNodes);
    adj = zeros(numNodes, numNodes);
    index = find(net ~= 0);
    adj(index) = 0.01 + (1-0.01).* rand(1,size(index,1)); %uniform distribution from [0.01, 1]
    save(['adj_' num2str(coef(i)) '_64.mat'], 'adj');
end
