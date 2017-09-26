-- this is the criterion that penalizes the distance between the predicted points and the surface tangent plane.
-- @shuhang
require 'nn'

local annoy = require 'annoy'

local SurfaceCriterion_kdtree, parent = torch.class('nn.SurfaceCriterion_kdtree', 'nn.Criterion')

function SurfaceCriterion_kdtree:__init(annPath, meshNormals, param)
   parent.__init(self)
   param = param or 0 
   self.subCriterion = nn.MSECriterion()   -- this is the criterion we used for computing the usual distance bewteen target and predicted points
   self.subCriterion.sizeAverage = false -- we deal with the batch size ourself.
   self.param = param
--   self.meshPoints = meshPoints
   self.meshNormals = meshNormals
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.sizeAverage = true
   self.meshTree = annoy.AnnoyIndex(3, 'euclidean') -- 3 means the 3 dimensional spaces our coordinates lie in
   -- load the kd tree we have already constructed
   --   self.meshTree:load('/Users/mark/Documents/VT_learning/data/meshTree.ann')
   self.meshTree:load(annPath)
end

function SurfaceCriterion_kdtree:updateOutput(input, y)

   local PerpendDistances = torch.Tensor(input:size(1))
   self.closeIndexes = torch.Tensor(input:size(1))
   self.gradSign = torch.Tensor(input:size(1)):fill(1)
   for i = 1, input:size(1) do
      local closeIndex = self.meshTree:get_nns_by_vector({input[i][1],input[i][2], input[i][3]},1)
      self.closeIndexes[i] = closeIndex[1] + 1 -- +1 because the index of kd tree starts with zero
      local vec = torch.Tensor(input[i]:size())
      if input:type() == 'torch.DoubleTensor' then 
	 vec = input[i] - torch.Tensor(self.meshTree:get_item_vector(closeIndex[1]))
      elseif input:type() == 'torch.CudaTensor' then
	 vec = vec:cuda()
	 vec = input[i] - torch.Tensor(self.meshTree:get_item_vector(closeIndex[1])):cuda()
      else
	 error('The input type to surfacecriterion_kdtree is unknown')
      end
      -- the perpendicular distance is calculated as a(x0-x) + b(y0-y) + c(z0-z) (assuming a, b, c has been normalized)
      local perpendDistance = torch.sum(torch.cmul(vec, self.meshNormals[self.closeIndexes[i]]))
      if perpendDistance < 0 then
	 self.gradSign[i] = -1
	 PerpendDistances[i] = perpendDistance * (-1)
      else 
	 PerpendDistances[i] = perpendDistance
      end
   end
   
   if self.sizeAverage then
      self.subDistance = self.subCriterion:forward(input, y) / input:size(1)
      self.out = self.subDistance + self.param * PerpendDistances:sum()/input:size(1)
   else
      self.subDistance = self.subCriterion:forward(input, y)
      self.out = self.subDistance + self.param * PerpendDistances:sum()
   end

   return self.out

end

function SurfaceCriterion_kdtree:project(input)

   local projected = torch.Tensor()
   if input:type() == 'torch.DoubleTensor' then
      projected:resizeAs(input)
   elseif input:type() == 'torch.CudaTensor' then
      projected = projected:cuda()
      projected:resizeAs(input)
   else
      error('The input type to SurfaceCriterion_kdtree is wrong')
   end
     
   for i = 1, input:size(1) do
      local closeIndex = self.meshTree:get_nns_by_vector({input[i][1],input[i][2], input[i][3]},1)
      local vec = torch.Tensor(input[i]:size())
      if input:type() == 'torch.DoubleTensor' then 
	 vec = input[i] - torch.Tensor(self.meshTree:get_item_vector(closeIndex[1]))
      elseif input:type() == 'torch.CudaTensor' then
	 vec = vec:cuda()
	 vec = input[i] - torch.Tensor(self.meshTree:get_item_vector(closeIndex[1])):cuda()
      else
	 error('The input type to surfacecriterion_kdtree is unknown')
      end
      projected[i] = input[i] - torch.mul(self.meshNormals[closeIndex[1]+1], torch.dot(vec, self.meshNormals[closeIndex[1]+1]))
   end
   return projected
end

function SurfaceCriterion_kdtree:updateGradInput(input, y)
   
   local subGrad = self.subCriterion:backward(input, y)
   
   local tangentGrad = torch.Tensor(input:size())
   if input:type() == 'torch.CudaTensor' then
      tangentGrad = tangentGrad:cuda()
   end
   for i =1, tangentGrad:size(1) do
      tangentGrad[i] = self.meshNormals[self.closeIndexes[i]]*self.gradSign[i]
   end
   
   if self.sizeAverage then
      self.gradInput = torch.div(subGrad, input:size(1)) + torch.div(tangentGrad:mul(self.param),input:size(1))
   else
      self.gradInput = subGrad +  tangentGrad:mul(self.param)
   end

   return self.gradInput 

end
   

   

   
