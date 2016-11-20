require 'nn'
require 'image'
local matio = require 'matio'
local dic = matio.load('ConvDict.mat').ConvDict.Dict[5]:double()
local img = image.load('lena.png', 1):div(255)
img = img:sub(1,30,1,30)
local numberOfKernels = 36
local kernelSize = 12
local csc = nn.Sequential()
-- split the 3d tensor into diferent slices
csc:add(nn.SplitTable(4))
-- construct a parallel table with a convolution kernal in each table element
local convolutions = nn.ParallelTable()
for i = 1, numberOfKernels do
   -- need to set weights with learn dictionary
   local convolutionKernel =  nn.SpatialConvolution(1, 1, kernelSize, kernelSize, (kernelSize - 1)/2, (kernelSize - 1)/2)
   convolutionKernel.weight:set(dic[{{}, {}, i}]:contiguous())
   convolutionKernel:noBias()
   convolutions:add(convolutionKernel)
end
csc:add(convolutions)
-- add each map together
csc:add(nn.CAddTable())
local x = torch.ones(1, 30, 30, 36)

csc:forward(x)
