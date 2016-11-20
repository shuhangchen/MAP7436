require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'gnuplot'
require 'nn'
package.path = package.path..";../bundleLevel/?.lua"
require 'fastAPL'
local matio = require 'matio'

local dic = matio.load('ConvDict.mat').ConvDict.Dict[5]
local img = image.load('lena.png', 1):div(255)

img = img:sub(1,30,1,30)

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


local feval = function (x)
   --- compute the loss functiona and gradient at given point
   -- 1/2 * \sum ||d*x -s||2 + \sum ||x||1
   
   -- check size of x
   assert(x:numel() == (img:size(1)+1)*(img:size(2)+1) * numberOfKernels, 'wrong input size of sparse codes')
   local xTensor = x:view(1, img:size(1) + 1, img:size(2) + 1, numberOfKernels)
   local pred = csc:forward(xTensor)
   local lossNet = criterion:forward(pred, img) 
   local gradCriterion = criterion:backward(pred, img)
   local inputGradNet = csc:backward(xTensor, gradCriterion)
   local l1grad = torch.sign(xTensor)
   local loss = 0.5 * lossNet + lambda * torch.norm(xTensor, 1)
   local grad = 0.5 * inputGradNet + lambda * l1grad
   return loss, grad:view(-1)
end

-- setup the parameter settings
local fValueTrack = {
   init = true,
   fTrack = torch.Tensor(1)
}
local ballCons = {
   radius = torch.norm(numberOfKernels * img) * 500, --
   center = torch.zeros((img:size(1)+1) * (img:size(2)+1) * numberOfKernels)
}

local opts = {
   ballConstraint = ballCons,
   beta = 0.65,
   theta = 0.7,
   lambda = 1e-2,
   initialLowerBound = 0,
   initialGap = 1e-3,
   maxUnconsIter = 100,
   maxIterPerFAPL = 150,
   maxIterPerFAPLgap = 500,
   maxBundles = 5,
   expanding = true,
   eps = 1e-5
}

local startTime = os.clock()
local xSolution = fastAPL.optimConstrained(feval, opts, fValueTrack)
local computeTime = os.clock() - startTime
print('time used is '..tostring(computeTime))
gnuplot.plot(fValueTrack.fTrack)
