#!/Users/shuhang/torch/install/bin/th

-- This file demonstartes the optimization of fast APL, a type of bundle level methods, with respect to the following unconstrained CP problem:
-- \min_x \|Ax - b\|_2^2

require 'torch'
require 'fastAPL'
require 'gnuplot'
sizeM = 20
sizeN = 20

local A = torch.randn(sizeM, sizeN);
local x = torch.randn(sizeN)
x = x:div(torch.norm(x))
local observationB = torch.mv(A, x)

local feval = function(xValue)
   local fValue = torch.pow(torch.norm(A*xValue - observationB), 2)
   local gradient = 2 * A:t() * (A * xValue - observationB) -- note, the transpose won't change the size of A itself.
   return fValue, gradient
end


local ballCons = {
   radius = 1,
   center = torch.zeros(sizeN)
}

local opts = {
   ballConstraint = ballCons,
   beta = 0.2,
   theta = 0.2,
   initialLowerBound = -1e-10,
   initialGap = 1e-3,
   maxUnconsIter = 10,
   maxIterPerFAPL = 150,
   maxIterPerFAPLgap = 500,
   maxBundles = 5,
   maxPhases = 200,
   expanding = true,
   eps = 1e-10,
}

local fValueTrack = {
   init = true,
   fTrack = torch.Tensor(1)
}

local xSolution = fastAPL.optimConstrained(feval, opts, fValueTrack)
local funValue = feval(xSolution)
local trueFvalue = feval(x)
print(funValue)
print(trueFvalue)
print(torch.pow(torch.norm(xSolution - x), 2))
gnuplot.plot(fValueTrack.fTrack)
