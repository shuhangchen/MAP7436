-- this is the criterion that penalizes the distance between the predicted points and the surface tangent plane.
-- @shuhang


local SurfaceCriterion, parent = torch.class('nn.SurfaceCriterion', 'nn.Criterion')

function SurfaceCriterion:__init(meshPoints, meshNormals, param)
   parent.__init(self)
   param = param or 0 
   self.subCriterion = nn.MSECriterion()   -- this is the criterion we used for computing the usual distance bewteen target and predicted points
   self.param = param
   self.meshPoints = meshPoints
   self.meshNormals = meshNormals
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.sizeAverage = true
end

function SurfaceCriterion:updateOutput(input, y)

   self.closeMeshPoints = torch.Tensor(input:size())
   self.closeMeshDistance = torch.Tensor(input:size(1), 1)
   self.perpendDistance = torch.Tensor(input:size(1))
   self.closeIndexes = torch.Tensor(input:size(1))
   for i = 1, input:size(1) do
      closeIndex = 1
      closeDistance = torch.sum(torch.pow(torch.sub(sell.meshPoints[closeIndex], input[i]), 2))
      for k = 2, self.meshPoints:size(1) do
	 distance = torch.sum(torch.pow(torch.sub(sell.meshPoints[k], input[i]), 2))
	 if distance < closeDistance do 
	    closeIndex = k
	    closeDistance = distance
	 end
      end
      self.closeMeshPoints[i] = self.meshPoints[closeIndex]:clone()
      self.closeMeshDistance[i] = closeDistance 
      self.closeIndexes[i] = closeIndex
      vec = input[i] - self.closeMeshPoints[i] 
      if torch.sum(torch.cmul(vec, self.meshNormals[i])) < 0 then
	 vec:mul(-1)
      end
      self.perpendDistance[i] = torch.sum(torch.cmul(vec, self.meshNormals[i]))
   end
   self.subDistance = self.subCriterion:forward(input, y)
  
   if self.sizeAverage then
      self.out = self.subDistance + self.perpendDistance:sum()/input:size(1)
   else
      self.out = self.subDistance + self.perpendDistance:sum()
   end
   return self.out

end

function SurfaceCriterion:updateGradInput(input, y)
   
   local subGrad = self.subCriterion:backward(input, y)
   
   local tangentGrad = torch.Tensor(input:size())

   for i =1, tangentGrad:size(1) do
      tangentGrad[i] = self.meshNormals[self.closeIndexes]
   end
   
   if self.sizeAverage then
      self.gradInput = subGrad + tangentGrad:div(input:size(1))
   else
      self.gradInput = subGrad + tangentGrad 
   end

   return self.gradInput 

end
   

   

   
