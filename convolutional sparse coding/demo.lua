-- demo for the convolutional sparse coding by convolutional ADMM

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'cbpdn'
require 'gnuplot'
local matio = require 'matio'

-- choose the learned dictionary with 36 filters with size of 12x12
local dic = matio.load('ConvDict.mat').ConvDict.Dict[5]
local img = image.load('lena.png', 1):div(255)
-- just for ease of testing
img = img:sub(1,30,1,30)
local lambda = 1e-2  -- the weighting parameter before the l-1 norm

local opts = {
   maxIter = 500,
   rho = 100 * lambda + 1,
   relax = 0.5,
   l1weight = 1,
   autoRho = false,
   autoRhoPeriod = 1,
   rhoRsdlRatio = 1.2,
   autoRhoScaling = true,
   rhoScaling = 1,
   rhoRsdlTarget = nil,
   noBndryCross = false,
   absStopTol = 0,
   relStopTol = 1e-4,
   noNegCoef = true,
   relStopTol = 1e-3
}

local startTime = os.clock()
local sparseCoef, objs = cbpdn.admm(dic, img, lambda, opts) -- the img here should be replaced by the low passed version
local computeTime = os.clock() - startTime
print('time used is '..tostring(computeTime))
gnuplot.plot(torch.Tensor(objs))
