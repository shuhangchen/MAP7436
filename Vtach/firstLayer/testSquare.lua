require 'torch'
require 'nn'
require 'cunn'
opt={
   hingeCoef = 0.5
}

local dataDim = 4
local dataNum = 10
local input = {}
input[1] = torch.rand(dataNum,dataDim)
input[2] = torch.rand(dataNum,dataDim)
local VTPAlabels = torch.cat(torch.ones(5,1):fill(-1),torch.ones(5,1),1)

local VTPACriterion = function(input, VTPAlabels)
   -- 0.5* (pairDistance_VT + pairDistance_PA)
   local mlp_PA = nn.PairwiseDistance(2)

   local distancePA = mlp_PA:forward({input[1], input[2]})
   errSqurePA = torch.cmul(distancePA, (torch.add(VTPAlabels,1):mul(0.5)))
--   print(errSqurePA)
   gradSquarePA = torch.Tensor(distancePA:size()):fill(1/10)
   mlp_PA:zeroGradParameters()
   errHinge = errSqurePA:mean()

   local gradOutputPA = mlp_PA:backward({input[1], input[2]}, gradSquarePA)
   
   -- multiply the coefficient into gradient vector
   gradHinge = {}
   gradHinge[1] =  gradOutputPA[1]:mul(opt.hingeCoef)
   gradHinge[2] =  gradOutputPA[2]:mul(opt.hingeCoef)
   return errHinge, gradHinge
end

local VTPACriterion_1 = function(input, VTPAlabels)
   -- 0.5* (pairDistance_VT + pairDistance_PA)
   local mlp_PA = nn.PairwiseDistance(2)
   local absCriterion = nn.AbsCriterion()
   local distancePA = mlp_PA:forward({input[1], input[2]})
   distancePA_filtered = torch.cmul(distancePA, (torch.add(VTPAlabels,1):mul(0.5)))
--   print(distancePA_filtered)
   errPA = absCriterion:forward(distancePA_filtered,torch.Tensor(distancePA:size()):zero())
   mlp_PA:zeroGradParameters()
   errHinge = errPA
   local gradAbsPA = absCriterion:backward(distancePA_filtered,torch.Tensor(distancePA:size()):zero())
   local gradOutputPA = mlp_PA:backward({input[1], input[2]}, gradAbsPA)
   
   -- multiply the coefficient into gradient vector
   gradHinge = {}
   gradHinge[1] =  gradOutputPA[1]:mul(opt.hingeCoef)
   gradHinge[2] =  gradOutputPA[2]:mul(opt.hingeCoef)
   return errHinge, gradHinge
end


local err,grad = VTPACriterion(input, VTPAlabels)
local err_1,grad_1 = VTPACriterion_1(input, VTPAlabels)
print(err)
print(grad[1])
print(err_1)
print(grad_1[1])
