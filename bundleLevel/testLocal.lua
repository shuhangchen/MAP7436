require 'torch'
require 'fastAPL'
require 'gnuplot'
sizeM = 20
sizeN = 40

local A = torch.randn(sizeM, sizeN);
local x = torch.randn(sizeN)
x = x:div(torch.norm(x))
local observationB = torch.mv(A, x)

local feval = function(xValue)
   local fValue = torch.pow(torch.norm(A*xValue - observationB), 2)
   local gradient = 2 * A:t() * (A * xValue - observationB) -- note, the transpose won't change the size of A itself.
   return fValue, gradient
end

local x1 = torch.randn(sizeN)
local x2 = torch.randn(sizeN)

local f1,g1 = feval(x1)
local f2,g2 = feval(x2)

local linearFunAtX = function(pot, fValue, grad)
   --- linear function is represented by a*x + b
   return function(evalPoint) return grad * evalPoint + fValue - grad * pot end
end

local pot, f, g = 1, 3, 2
print(linearFunAtX(pot, 3, 2)(3))
