require 'torch'
-- torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'gnuplot'
require 'nn'
package.path = package.path..";../bundleLevel/?.lua"
require 'fastUSL'
local matio = require 'matio'

local dic = matio.load('ConvDict.mat').ConvDict.Dict[5]:double()
local img = image.load('lena.png', 1):div(255)

img = img:sub(1,18,1,18)

local lambda = 1e-2

-- construct an convolution network for computing the objective function
local numberOfKernels = dic:size(3)
local kernelSize = dic:size(1)
local csc = nn.Sequential()
-- split the 3d tensor into diferent slices
csc:add(nn.SplitTable(4))
-- construct a parallel table with a convolution kernal in each table element
local convolutions = nn.ParallelTable()
for i = 1, numberOfKernels do
   -- need to set weights with learn dictionary
   local convolutionKernel =  nn.SpatialConvolution(1, 1, kernelSize, kernelSize,1.0 , 1.0 , (kernelSize - 1)/2, (kernelSize - 1)/2)
   convolutionKernel.weight:set(dic[{{}, {}, i}]:contiguous())
   convolutionKernel:noBias()
   convolutions:add(convolutionKernel)
end
csc:add(convolutions)
-- add each map together
csc:add(nn.CAddTable())
-- define a criterion for network and compute the l1 loss
local criterion = nn.MSECriterion()
criterion.sizeAverage = false


local feval = function (x, eta)
   --- compute the loss functiona and gradient at given point
   -- 1/2 * \sum ||d*x -s||2 + \sum ||x||1
   
   -- check size of x
   assert(x:numel() == (img:size(1)+1)*(img:size(2)+1) * numberOfKernels, 'wrong input size of sparse codes')
   local xTensor = x:view(1, img:size(1) + 1, img:size(2) + 1, numberOfKernels)
   local pred = csc:forward(xTensor)
   local lossNet = criterion:forward(pred, img) 
   local gradCriterion = criterion:backward(pred, img)
   local inputGradNet = csc:backward(xTensor, gradCriterion)
   if eta == 0 then
      local l1grad = torch.sign(xTensor)
      local loss = 0.5 * lossNet + lambda * torch.norm(xTensor, 1)
      local grad = 0.5 * inputGradNet + lambda * l1grad
      return loss, grad:view(-1), loss
   else
      local dualx = xTensor / eta    -- by the maximization condition of dual problem
      -- then we need to normalize sparse coefficient for each dictionary element -- project the outsider inside the hypercube
      for i = 1, dualx:size(3) do
	 local slice = dualx[{{},{},i}]
	 if slice:max() > 1 then
	    slice:div(slice:max())
	 end
      end
      --!!!!!!!!!!! still need to check the matlab script of matlab TV
      local loss = 0.5 * lossNet + lambda * (torch.dot(xTensor, dualx) -0.5 * eta * math.pow(torch.norm(dualx), 2))  -- the smoothing term
      local grad = 0.5 * inputGradNet + lambda * dualx  -- replace the smooth term
      local loss0 = lossNet + lambda * torch.norm(xTensor,1)
      return loss, grad:view(-1), loss0
   end
end

local fevalNOgrad = function(x, eta)
   --- compute the loss functiona and gradient at given point
   -- 1/2 * \sum ||d*x -s||2 + \sum ||x||1
   
   -- check size of x
   assert(x:numel() == (img:size(1)+1)*(img:size(2)+1) * numberOfKernels, 'wrong input size of sparse codes')
   local xTensor = x:view(1, img:size(1) + 1, img:size(2) + 1, numberOfKernels)
   local pred = csc:forward(xTensor)
   local lossNet = criterion:forward(pred, img) 
   if eta == 0 then
      local l1grad = torch.sign(xTensor)
      local loss = 0.5 * lossNet + lambda * torch.norm(xTensor, 1)
      return loss, loss
   else
      local dualx = xTensor / eta    -- by the maximization condition of dual problem
      -- then we need to normalize sparse coefficient for each dictionary element -- project the outsider inside the hypercube
      for i = 1, dualx:size(3) do
	 local slice = dualx[{{},{},i}]
	 if slice:max() > 1 then
	    slice:div(slice:max())
	 end
      end
      --!!!!!!!!!!! still need to check the matlab script of matlab TV
      local loss = 0.5 * lossNet + lambda * (torch.dot(xTensor, dualx) -0.5 * eta * math.pow(torch.norm(dualx), 2))  -- the smoothing term
      local loss0 = lossNet + lambda * torch.norm(xTensor,1)
      return loss, loss0
   end
end

local function checkgrad(opfunc, x, eta)
   local _, dC, _ = opfunc(x, eta)
   dC:resize(x:size())

   -- compute the approximate numerical gradient
   local eps = (x:type() == 'torch.DoubleTensor') and 1e-6 or 1e-3
   local dC_est = torch.Tensor(dC:size())
   for i = 1, dC:size(1) do
      x[i] = x[i] + eps
      local C1 = opfunc(x, eta)
      x[i] = x[i] - 2*eps
      local C2 = opfunc(x, eta)
      x[i] = x[i] + eps
      dC_est[i] = (C1 - C2) / (2 * eps)
   end
   local diff = torch.norm(dC_est - dC)/torch.norm(dC_est + dC)
   return diff, dC, dC_est
end

print(' checking the gradients now')
local input = torch.rand((img:size(1)+1)*(img:size(2) + 1)* numberOfKernels)
local eta = 0
local diff, dC, dC_est = checkgrad(feval, input, eta)
print(diff)

