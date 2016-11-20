-- this file reads the provides data of ground truth network data and (simulated?) cascade data

require 'torch'
require 'pl.stringx'
local npy4th = require 'npy4th'

readData = {} 
local function readNetwork(networkName, numNodes)
   -- declare two types of data
   local nodes
   local edges
   -- read the network data
   local file = io.open(networkName, 'r')
   if file then
      for line in file:lines() do
	 local lineVec = line:split(',')
	 if #lineVec ~=1 and lineVec[1] ~= '\n' then
	    if #lineVec == 2 then
	       if nodes == nil then
		  nodes = torch.Tensor(lineVec):view(1,-1)
	       else
		  nodes = torch.cat(nodes, torch.Tensor(lineVec):view(1, -1), 1)
	       end
	    elseif #lineVec == 3 then
	       if edges == nil then
		  edges = torch.Tensor(lineVec):view(1,-1)
	       else
		  edges = torch.cat(edges, torch.Tensor(lineVec):view(1, -1), 1)
	       end
	    else
	       error(' we are expecting different input file format')
	    end
	 end
      end
   end
   io.close(file)
   local adjMatrix = torch.zeros(numNodes, numNodes)
   for i = 1, edges:size(1) do
      adjMatrix[edges[i][1] + 1][edges[i][2] + 1] = edges[i][3]
   end
   return adjMatrix
end

local function readCascades(cascadesName, numNodes)
   -- each element in the cascade table is an one dimensional vector
   local nodes
   local cascades = {}
   local numCascades = 1
   -- read in cascade data
   local file = io.open(cascadesName, 'r')
   if file then
      for line in file:lines() do
	 local lineVec = line:split(',')
	  if #lineVec ~=1 and lineVec[1] ~= '\n' then
	    if #lineVec == 2 then
	       if nodes == nil then
		  nodes = torch.Tensor(lineVec):view(1,-1)
	       else
		  nodes = torch.cat(nodes, torch.Tensor(lineVec):view(1, -1), 1)
	       end
	    else
	       cascades[numCascades] = torch.Tensor(lineVec)
	       numCascades = numCascades + 1
	    end
	  end
      end
   end
   io.close(file)
   -- construct the cascade matrix
   local cascadeMatrix = torch.Tensor(#cascades, numNodes):fill(-1)
   for i = 1, #cascades do
      for j = 1, cascades[i]:size(1), 2 do
	 cascadeMatrix[i][cascades[i][j] + 1] = cascades[i][j+1]
      end
   end
   return cascadeMatrix
end


function readData.read(netFile, cascadeFile, numNodes)
   return readNetwork(netFile, numNodes), readCascades(cascadeFile, numNodes)
end
