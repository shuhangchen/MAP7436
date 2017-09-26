local annoy = require "annoy"

-- t:add_item(1, {0,0,0})
-- t:add_item(2, {1.0,1.2,2.0})
-- t:build(10) -- 10 trees
-- t:save('test.ann')

-- local test = {1.0, 2.0, 3.0}

-- print(t:get_nns_by_vector(test, 1))


local dim = 3

-- very important
local meshTree = annoy.AnnoyIndex(dim, 'euclidean')  -- works like an empty table ?

local meshes = torch.load('/home/mark/Documents/VT_learning/data/meshCentroids.t7')

local numMeshPoints = meshes.mean_coordinates:size(1)

-- it has to start with zero!!!
for i= 0, numMeshPoints-1 do
   local v = {meshes.mean_coordinates[i+1][1], meshes.mean_coordinates[i+1][2], meshes.mean_coordinates[i+1][3]}
   meshTree:add_item(i, v)
end

meshTree:build(10)   -- use 10 trees
meshTree:save('/home/mark/Documents/VT_learning/data/meshTree.ann')

-- brutal force search version
local closestMesh= function(vector)
   local closeIndex = 1
   local closeDistance = torch.sum(torch.pow(meshes.mean_coordinates[closeIndex] - vector, 2))
   for k = 2, meshes.mean_coordinates:size(1) do
      distance = torch.sum(torch.pow(meshes.mean_coordinates[k]- vector, 2))
      if distance < closeDistance then
	 closeIndex = k
	 closeDistance = distance
      end
   end
   return closeIndex
end

-- test the model with brutal force method

local input = torch.rand(10,3):mul(10)

for i = 1, input:size(1) do
   index_true = closestMesh(input[i])
   index_ann = meshTree:get_nns_by_vector({input[i][1],input[i][2], input[i][3]},1)
--   print(index_ann)
--   print('the correct one is '..tostring(index_true)..' and the ann one is '..tostring(index_ann[1] + 1))
end

vec = meshTree:get_item_vector(2)
print(meshes.mean_coordinates[2])
print(vec)
print(torch.Tensor(vec))




