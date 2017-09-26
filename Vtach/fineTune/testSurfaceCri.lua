-- this script tests the surface constraint 

-- @shuhang

require 'torch'
require 'nn'
-- require 'cunn'
require 'SurfaceCriterion'
require 'SurfaceCriterion_kdtree'
local annoy= require 'annoy'
-- input = torch.Tensor({{0.6256, 0.1132, 0.1355}, {0.4758, 0.2045, 0.3783}})

-- target = torch.Tensor({{0.9695, 0.9498, 0.7337},{ 0.2608,  0.9029,  0.8267}})


-- meshPoints = torch.Tensor({{0.0074,  0.3840,  0.8664}, {0.4852,  0.3037,  0.3290}, { 0.3080,  0.4313,  0.3479}, {0.7200,  0.7089,  0.5806}})

-- meshNormals = torch.Tensor({{0.3092, 0.9049, 0.2924}, {0.4170,  0.2211,  0.8816}, {0.7788,  0.2862,  0.5581}, {0.9618,  0.2738, 0.0026}})

-- -- for i=1, meshNormals:size(1) do
-- --   meshNormals[i]:div(torch.norm(meshNormals[i]))
-- -- end

-- criterion = nn.SurfaceCriterion(meshPoints, meshNormals, 0.5)


-- local output = criterion:forward(input, target)

-- local df = criterion:backward(input, target)

-- the function value of the SurfaceCriterion is checked and passed, now we check the gradients it gives us


local input = torch.rand(20, 3)

local target = torch.rand(20,3)

local meshPoints = torch.rand(50,3)
local meshNormals = torch.rand(50,3)
-- constructs the kd tree index system and save it
local annPath= '/Users/mark/Documents/VT_learning/data/test.ann'
local annTest = annoy.AnnoyIndex(3, 'euclidean') --, 'euclidean')
for i=0, meshPoints:size(1) - 1 do
   local vec = {meshPoints[i+1][1], meshPoints[i+1][2], meshPoints[i+1][3]}
   annTest:add_item(i, vec)
end

annTest:build(50)
annTest:save(annPath)

-- make sure the mesh normal vector has unit length
for i=1, meshNormals:size(1) do
   meshNormals:div(torch.norm(meshNormals[i]))
end

local feval = function(x)
   collectgarbage()
   local loss = criterion:forward(x, target)
   local grad = criterion:backward(x, target)
   return loss, grad
end

function checkgrad(opfunc, x)
   local _, grad = opfunc(x)
   grad:resize(x:size())
   -- the below input should be x, my bad, though it does not make big difference
   -- compute the numerical approximation gradient
   local eps = (x:type() == 'torch.DoubleTensor') and 1e-6 or 1e-3 
   local grad_test = torch.DoubleTensor(grad:size())
   for i = 1, grad:size(1) do
      for j = 1, grad:size(2) do
	 x[i][j] = x[i][j] + eps
	 local f1 = opfunc(x)
	 x[i][j] = x[i][j] - 2*eps
	 local f2 = opfunc(x)
	 x[i][j] = x[i][j] + eps
	 grad_test[i][j] = (f1 - f2) / (2 * eps)
      end
   end
   local df= torch.norm(torch.add(grad,-1, grad_test)) / torch.norm(torch.add(grad, grad_test))
   return df
end

-- df = checkgrad(feval, input)

-- print(df)


-- now we compare the results between brutal force criterion and kd tree based surface criterion

criterion_true = nn.SurfaceCriterion(meshPoints, meshNormals, 1)
criterion = nn.SurfaceCriterion_kdtree(annPath, meshNormals, 1)
loss_true = criterion_true:forward(input, target)
loss_ann = criterion:forward(input, target)
grad_true = criterion_true:backward(input, target)
grad_ann = criterion:backward(input, target)
print(loss_true)
print(loss_ann)
print(grad_true)
print(grad_ann)
