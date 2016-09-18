
-- this file tests whether the implementation of TV norm calculation and gradient evaluation is right, compared to the matlab codes provided by Wei Zhang
require 'torch'
require 'TVeval'
local matio = require 'matio'
--
local imgSize = 30
local img = torch.rand(imgSize, imgSize):mul(10)
img = img:view(img:nElement())
local A = torch.randn(imgSize * imgSize / 4, imgSize * imgSize)
local b = torch.mv(A, img)
local eta = 0
local lambda = 1000
local x = torch.rand(imgSize * imgSize)
local f, grad, f0 = TVeval.feval(A, x, b, eta, lambda, imgSize * imgSize , imgSize, imgSize)
matio.save('TVTest.mat',{A = A, x = x, b = b, f = f, grad = grad, f0 = f0, diffMatrix = diffMatrix, diffOfDiffMatrix = diffOfDiffMatrix})




