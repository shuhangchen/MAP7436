-- this is the criterion that penalizes the distance between the predicted points and the surface tangent plane.
-- @shuhang
require 'nn'

local SurfaceCriterion, parent = torch.class('nn.SurfaceCriterion', 'nn.Criterion')

function SurfaceCriterion:__init(meshPoints, meshNormals, param)
   parent.__init(self)
   param = param or 0 
   self.subCriterion = nn.MSECriterion()   -- this is the criterion we used for computing the usual distance bewteen target and predicted points
   self.subCriterion.sizeAverage = false -- we deal with the batch size ourself.
   self.param = param
   self.meshPoints = meshPoints
   self.meshNormals = meshNormals
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.sizeAverage = true
end

function SurfaceCriterion:updateOutput(input, y)

   local PerpendDistances = torch.Tensor(input:size(1))
   self.closeIndexes = torch.Tensor(input:size(1))
   self.gradSign = torch.Tensor(input:size(1)):fill(1)
   for i = 1, input:size(1) do
      closeIndex = 1
      closeDistance = torch.sum(torch.pow(self.meshPoints[closeIndex] - input[i], 2))
      for k = 2, self.meshPoints:size(1) do
	 distance = torch.sum(torch.pow(self.meshPoints[k]- input[i], 2))
	 if distance < closeDistance then
	    closeIndex = k
	    closeDistance = distance
	 end
      end
      self.closeIndexes[i] = closeIndex
      vec = input[i] - self.meshPoints[self.closeIndexes[i]] 
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

function SurfaceCriterion:updateGradInput(input, y)
   
   local subGrad = self.subCriterion:backward(input, y)
   
   local tangentGrad = torch.Tensor(input:size())

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
   

   

   
