#!/Users/shuhang/torch/install/bin/th

-- this file demonstartes the optimization and performance of fast USL method, a bundle level type algorithm for solving convex optimization algorithms, on TV regularized MR image reconstruction
require 'torch'
require 'image'
require 'TVeval'
require 'fastUSL'
require 'gnuplot'

local img = image.load('brainSmall.png', 1):select(1,1)
local sizeM, sizeN= img:size(1), img:size(2)
local N = sizeN * sizeM
local nSample = N / 4
local sigma = 1e-3   -- standrad deviation of noise 
-- generate projection matrix
local A = torch.randn(nSample, N)

local b = A * img:view(img:nElement()) + torch.randn(nSample) * sigma
local ballCons = {
   radius = torch.norm(img:view(img:nElement())) * 1.2,
   center = torch.zeros(N)
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

local feval = function (x, eta)
   return TVeval.feval(A, x, b, eta, opts.lambda, N, sizeM, sizeN) 
end

local fevalNOgrad = function(x, eta)
   return TVeval.fevalNOgrad(A, x, b, eta, opts.lambda, N, sizeM, sizeN)
end

local fValueTrack = {
   init = true,
   fTrack = torch.Tensor(1)
}

local startTime = os.clock()
local xSolution = fastUSL.optimConstrained(feval, fevalNOgrad, opts, fValueTrack)
xSolution = xSolution:view(sizeM, sizeN)
image.save('recBrain.png', xSolution)
local computeTime = os.clock() - startTime
print('time used is '..tostring(computeTime))
gnuplot.plot(fValueTrack.fTrack)
