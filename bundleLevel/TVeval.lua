require 'torch'

TVeval = {}

local function diffMatrix(img)
   -- compute the difference of x, y directions
   local diffx = torch.cat(img[{{2,-1},{}}] - img[{{1,-2},{}}], torch.zeros(1, img:size(2)),1)
   local diffy = torch.cat(img[{{},{2,-1}}] - img[{{},{1,-2}}], torch.zeros(img:size(1), 1), 2)
   local diffMatrix = torch.cat(diffx, diffy, 3)
   return diffMatrix
end

local function diffOfDiffMatrix(diff)
   --- compute the TV norm of an image
   local diffXpadding = torch.cat({torch.zeros(1, diff:size(2)), diff[{{1, -2},{},{1}}], torch.zeros(1, diff:size(2))}, 1)
   local diffx = diffXpadding[{{2,-1},{}}] - diffXpadding[{{1,-2},{}}]
   local diffYpadding = torch.cat({torch.zeros(diff:size(1), 1), diff[{{}, {1,-2},{2}}], torch.zeros(diff:size(1), 1)}, 2)
   local diffy = diffYpadding[{{},{2,-1}}] - diffYpadding[{{},{1,-2}}]
   return -diffx - diffy
end

function TVeval.feval(A, x, b, eta, lambda, N, sizeM, sizeN)
   -- compute the objective function and gradient of TV regularized least squares (PPI)
   local f, grad, f0
   local img = x:view(sizeM, sizeN)
   local diffTensor = diffMatrix(img)
   local diffVec= diffTensor:clone():view(-1, 1)
   local pointTVNorm = torch.sqrt(torch.pow(diffTensor[{{},{},{1}}],2) + torch.pow(diffTensor[{{},{},{2}}],2))
   local TVnorm = pointTVNorm:sum()
   local residual = (A * x - b) 
   
   if eta == 0 then
      -- only TV + data fidality
      f = 0.5 * math.pow(torch.norm(residual), 2) + lambda * TVnorm
      pointTVNorm[pointTVNorm:eq(0)] = 1
      diffTensor[{{},{},{1}}]:cdiv(pointTVNorm)
      diffTensor[{{},{},{2}}]:cdiv(pointTVNorm)
      local gradTV = diffOfDiffMatrix(diffTensor)
      gradTV = gradTV:view(-1, 1)
      grad = A:t() * residual + lambda * gradTV
      f0 = f
   else
      diffTensor:div(eta)
      pointTVNorm = torch.sqrt(torch.pow(diffTensor[{{},{},{1}}],2) + torch.pow(diffTensor[{{},{},{2}}], 2))
      pointTVNorm[pointTVNorm:lt(1)] = 1
      diffTensor[{{},{},{1}}]:cdiv(pointTVNorm)
      diffTensor[{{},{},{2}}]:cdiv(pointTVNorm)
      local diffVecEta = diffTensor:view(-1, 1)
      f = 0.5 * math.pow(torch.norm(residual), 2) + lambda * (torch.dot(diffVec, diffVecEta) - 0.5 * eta * math.pow(torch.norm(diffVecEta), 2))
      local gradTV = diffOfDiffMatrix(diffTensor)
      gradTV = gradTV:view(-1, 1)
      grad = A:t() * residual + lambda * gradTV
      f0 = 0.5 * math.pow(torch.norm(residual), 2) + lambda * TVnorm
   end
   return f, grad, f0 
end

function TVeval.fevalNOgrad(A, x, b, eta, lambda, N, sizeM, sizeN)
   -- compute only the objective function of TV regularized least squares (PPI)
   local f, f0
   local img = x:view(sizeM, sizeN)
   local diffTensor = diffMatrix(img)
   local diffVec= diffTensor:clone():view(-1, 1)
   local pointTVNorm = torch.sqrt(torch.pow(diffTensor[{{},{},{1}}],2) + torch.pow(diffTensor[{{},{},{2}}],2))
   local TVnorm = pointTVNorm:sum()
   local residual =  (A * x - b) 
   if eta == 0 then
      -- only TV + data fidality
      f = 0.5 * math.pow(torch.norm(residual), 2) + lambda * TVnorm
      f0 = f
   else
      diffTensor:div(eta)
      pointTVNorm = torch.sqrt(torch.pow(diffTensor[{{},{},{1}}],2) + torch.pow(diffTensor[{{},{},{2}}], 2))
      pointTVNorm[pointTVNorm:lt(1)] = 1
      diffTensor[{{},{},{1}}]:cdiv(pointTVNorm)
      diffTensor[{{},{},{2}}]:cdiv(pointTVNorm)
      local diffVecEta = diffTensor:view(-1, 1)
      f = 0.5 * math.pow(torch.norm(residual), 2) + lambda * (torch.dot(diffVec, diffVecEta) - 0.5 * eta * math.pow(torch.norm(diffVecEta), 2))
      f0 = 0.5 * math.pow(torch.norm(residual), 2) + lambda * TVnorm
   end
   return f, f0
end
