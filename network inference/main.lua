require 'torch'
require 'readData'
local npy4th = require 'npy4th'
---- this is for test of above two functionality
local netfileName = 'netrate/kronecker-core-periphery-n1024-h10-r0_01-0_25-network.txt'
local casfileName = 'netrate/kronecker-core-periphery-n1024-h10-r0_01-0_25-1000-cascades.txt'
local numNodes = 1024
local horizon = 10
local adj, cascade = readData.read(netfileName, casfileName, numNodes)

local numCascades = torch.zeros(numNodes)
local phi_2 = torch.zeros(adj:size())
local phi_1 = torch.zeros(adj:size())

for c = 1, cascade:size(1) do
   local index01 = cascade[c]:ne(-1)
   local idx = torch.nonzero(index01):view(-1)
   local sorted, order = torch.sort(cascade[c]:maskedSelect(index01))
   for i = 2, sorted:size(1) do
      numCascades[idx[order[i]]] = numCascades[idx[order[i]]] + 1
      for j = 1, i - 1 do
	 -- for simiplicity, we just use exponential distribution here
	 -- node j -> node i, log survival
	 phi_2[idx[order[j]]][idx[order[i]]] = phi_2[idx[order[j]]][idx[order[i]]] + sorted[i] - sorted[j]
      end
   end

   for j = 1, numNodes do
      -- find all survival nodes
      if index01[j] == 0 then
	 for i = 1, sorted:size(1) do
	    -- use the exponential distribution
	    phi_1[idx[order[i]]][j] = phi_1[idx[order[i]]][j] + (horizon - sorted[i])
	 end
      end
   end
end


-- having computed the phi_1 and phi_2, we save them along with data for CVX PYTHON to use later
-- it is a little messy since I don't know whether it can save structured variable
npy4th.savenpy('data/adj.npy', adj)
npy4th.savenpy('data/cascade.npy', cascade)
npy4th.savenpy('data/phi_1.npy', phi_1)
npy4th.savenpy('data/phi_2.npy',phi_2)
npy4th.savenpy('data/numNodes.npy', torch.Tensor({numNodes}))
npy4th.savenpy('data/numCascades.npy', numCascades)
