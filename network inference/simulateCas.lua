-- this script receive the file name for reading adjaceny matrix and simulate the generation of cascades

local matio = require 'matio'
require 'torch'


local function activateNode(node, neighborNode,  alpha_ij, cascade, horizon, activationHistroy)
   -- sample the activation time based on exponential distribution of alpha_ij
   activationHistroy[node][neighborNode] = 1
   local timeInterval = torch.exponential(alpha_ij)
   if timeInterval + cascade[node] <= horizon then
      cascade[neighborNode] = timeInterval + cascade[node]
      return true
   else
      return false
   end
end

local function activateNeighborOfNode(node, adj, horizon, cascade, activationHistroy)
   -- row i consists of the elements alpha_i,j
   -- take care the indexs found by nonzero here is two dimensional
   local neighborsByteIndex = adj[node]:gt(0):nonzero()
   if neighborsByteIndex:nElement() == 0 then
      return false
   end
   local neighborsOfNode = neighborsByteIndex[{{}, 1}]
   local success = false
   for i = 1, neighborsOfNode:nElement() do
      -- check whether this node has been activtaed, and this activation has not been performed
      if cascade[neighborsOfNode[i]] < 0 and activationHistroy[node][neighborsOfNode[i]] == 0 then
	 local onepass = activateNode(node, neighborsOfNode[i], adj[node][neighborsOfNode[i]], cascade, horizon, activationHistroy)
	 success = (success or onepass)
      end
   end
   return success
end

local function runOneEpisode(sourceIndex, adj, horizon, cascade, activationHistroy)
   -- run one episode
   -- take all active nodes and sample their activation time
   local success = false
   for i = 1, sourceIndex:nElement() do
      local onepass = activateNeighborOfNode(sourceIndex[i], adj, horizon, cascade, activationHistroy)
      success = (success or onepass)
   end
   return success
end

local function activateNet(adj, horizon, cascade, activationHistroy)
   -- activate the neighbors of source nodes in sourceIndex in one cascade
   local maxEpisode = 1000
   local episode = 1
   local success = false
   local successiveStucks = 0
   local maxStucks = 5
   while (episode <= maxEpisode and successiveStucks < maxStucks) do
      -- find the active nodes
      -- take care, the indexs found by nonzero here is two dimensional
      local sourceIndex = cascade:ge(0):nonzero()[{{}, 1}]
      success = runOneEpisode(sourceIndex, adj, horizon, cascade, activationHistroy)
      if not success then
	 successiveStucks = successiveStucks + 1
      else
	 episode = episode + 1
	 successiveStucks = 0
      end
   end
end


local function simulate(adjName, cascadeName, horizon, numCascades)
   -- generate the cascades
   local adj = torch.load(adjName)
   local numNodes = adj:size(1)
   local cascades = torch.Tensor(numCascades, numNodes):fill(-1)

   for c = 1, numCascades do
      -- for each cascade, randomly draw the source nodes
      local sourceIndex = torch.random(1, numNodes)
      -- set the activation time for sources nodes to be zero
      cascades[c][sourceIndex] = 0
      -- activate the network given the source nodes
      local activationHistroy = torch.ByteTensor(numNodes, numNodes):fill(0)
      activateNet(adj, horizon, cascades[c], activationHistroy)
   end
   torch.save(cascadeName, cascades) 
end


-- main script
-- local adjName = 'simulation_data/adj_renyi_1_1024.t7'
-- local numCascades = 1
-- local cascadeName = 'simulation_data/cascade_renyi_1_'..tostring(numCascades)..'.t7'

local adjName = 'simulation_data/adj_renyi_2_1024.t7'
local numCascades = 1000
local cascadeName = 'simulation_data/cascade_renyi_2_'..tostring(numCascades)..'.t7'
local horizon = 10

simulate(adjName, cascadeName, horizon, numCascades)


