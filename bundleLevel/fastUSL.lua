-- This file implements the fast USL algorithm, a type of bundle level method for optimizing CP problem, with function evaluation returining function value and gradient (like first order oracle)

require 'torch'
require 'dualSolver'
require 'permuteMatrix'

fastUSL = {}

local linearFunAtX = function(pot, fValue, grad)
   --- linear function is represented by a*x + b
   return function(evalPoint) return grad * evalPoint + fValue - grad * pot end
end

local function proxMapping_kkt(x, bundles, proj_optimal_condition, new_cuttingPlane, proxCenter, bundleLevel)
   -- returns exact solution for proximal mapping
   -- namely eq(3.4) x_k = \arg\min_{x\in Q_k} 0.5 * \|x - x_bar\|^2
   local totalBundle = {}
   if bundles.size == 0 then
      -- in this case, both proj_optimal and new_cutting .coef_b are vectors, so we concatenate them in the 2rd dimension
      totalBundle.coef_A = torch.cat(proj_optimal_condition.coef_a:view(1,-1), new_cuttingPlane.coef_a:view(1,-1), 1)
      totalBundle.coef_b = torch.cat(proj_optimal_condition.coef_b, bundleLevel - new_cuttingPlane.coef_b)
   else
      totalBundle.coef_A = torch.cat({bundles.coef_A[{{1, bundles.size}, {}}], proj_optimal_condition.coef_a:view(1,-1), new_cuttingPlane.coef_a:view(1,-1)}, 1)
      totalBundle.coef_b = torch.cat({bundleLevel - bundles.coef_b[{{1, bundles.size}}], proj_optimal_condition.coef_b, bundleLevel - new_cuttingPlane.coef_b})
   end

   local M = totalBundle.coef_A * totalBundle.coef_A:t()
   M = torch.cat(M, -torch.eye(bundles.size + 2), 2)
   -- + 2 is for optimal condtion and new cutting plane
   local c = torch.mv(totalBundle.coef_A, proxCenter) - totalBundle.coef_b
   local minDual, solutionErr = dualSolver.solve(M, c, fastUSL.permuteTable[M:size(1) -1])
   if torch.norm(minDual) >= 0 and solutionErr < 1e-5 then
      return proxCenter - torch.mv(totalBundle.coef_A:t(), minDual[{{1,bundles.size + 2}}])
   else
      return x
   end
end



local function gapReduction(feval, fevalNOgrad, proxCenter, Q, lowerBound, upperBound, bundles, opt, radius)
   -- the gap reduction procedure for fastAPL

   -- define the generation of alpha sequence
   local alpha_sequence =function (iteration) return  2 / (iteration +1) end

   -- initilization
   local iter = 0
   local alpha
   local f_upperBound_0 = upperBound
   local f_upperBound = upperBound
   local bundleLevel = opt.beta * lowerBound + (1 - opt.beta) * f_upperBound_0
   local eta = opt.theta * (f_upperBound_0 -  bundleLevel)/ (2 * Q)
   local x_u = proxCenter:clone()
   local x_seq = proxCenter:clone()

   -- not sure about this one, the original paper says it is the ball center, but the code and APL paper said it is this initialization
   local x_l, f_x_l, grad_f_x_l, f_x_l_eta

   -- print(x_seq:size())
   -- setup constraint defined by bundles (linear functions)

   -- optimal condition for projection
   local proj_optimal_condition = {
      coef_a = torch.Tensor(x_seq:size()):fill(0),
      coef_b = torch.Tensor(1):fill(0)
   }
   
   while iter < opt.maxIterPerFAPLgap and ((f_upperBound - lowerBound) > opt.eps)
   do
      iter = iter + 1
      alpha = alpha_sequence(iter)
      x_l = (1 - alpha) * x_u + alpha * x_seq
      f_x_l, grad_f_x_l, f_x_l_eta = feval(x_l, eta)
      -- given the new linear function h(x_l, x), we update the bundle with eq(3.3) in the original paper
      -- the coef_b here only contains the cutting plane, no bundle level in eq(3.3)
      local new_cuttingPlane = {
	 coef_a = grad_f_x_l:clone(), 
	 coef_b = torch.Tensor({f_x_l - torch.dot(grad_f_x_l, x_l)})
      }
      
      x_seq = proxMapping_kkt(x_seq, bundles, proj_optimal_condition, new_cuttingPlane, proxCenter, bundleLevel)
      
      if torch.norm(x_seq) > radius then
	 -- let us do what step 2 in gap reduction of the paper says if proxMapping does not have a solution
	 -- reset the bundles if there are no solutions
	 bundles.coef_A:zero()
	 bundles.coef_b:zero()
	 bundles.size = 0
	 return x_u, bundleLevel, f_upperBound, x_u:clone(), Q
      end

      -- otherwise we update the upper bound,
      -- (assuming the inner expression of last if-statement ends with return or throw expeption
      local x_u_hat = (1- alpha) * x_u + alpha * x_seq
      local f_u_hat, f_u_hat_eta = fevalNOgrad(x_u_hat, eta)
      if f_u_hat_eta < f_upperBound then
	 x_u = x_u_hat
	 f_upperBound = f_u_hat_eta
      end


      if (f_upperBound <= bundleLevel + opt.theta * (f_upperBound_0 - bundleLevel)) then
	 return x_u, lowerBound, f_upperBound, x_u:clone(), Q
      elseif (f_u_hat <= bundleLevel + 0.5 * opt.theta * (f_upperBound_0 - bundleLevel)) then
	 return x_u, lowerBound, f_upperBound, x_u:clone(), 2 * Q
      end

      -- update the bundle
      bundles.coef_A = torch.cat(new_cuttingPlane.coef_a:view(1,-1), bundles.coef_A[{{1, opt.maxBundles - 1},{}}], 1)
      bundles.coef_b = torch.cat(new_cuttingPlane.coef_b, bundles.coef_b[{{1, opt.maxBundles - 1}}], 1)
      if bundles.size < opt.maxBundles then
	 bundles.size = bundles.size + 1
      end
      -- update the proj_optimal condtion
      proj_optimal_condition.coef_a = proxCenter - x_seq
      proj_optimal_condition.coef_b[1] = torch.dot(proj_optimal_condition.coef_a, x_seq)
      collectgarbage()      
   end

   return x_u, bundleLevel, f_upperBound, x_u:clone(), Q
end


function fastUSL.optim(feval, fevalNOgrad, opts, radius, Q, proxCenter, lowerBound, upperBound, fValueTrack)
   --- ball constrained bundle level methods called fastAPL, minimizing feval function

   local bundles = {
      -- polyhedral defined by linear inequality constraints
      -- each row of coef_A is a vector, called A_i
      -- each entry of the vector coef_b is denoted b_i
      -- each linear inequality is defined as <A_i, x > <= b_i
      coef_A = torch.Tensor(opts.maxBundles, proxCenter:size(1)):zero(),
      coef_b = torch.Tensor(opts.maxBundles):zero(),
      size = 0
   }
   local hat_x
   local iter = 0
   while ((upperBound - lowerBound) > opts.eps and iter < opts.maxIterPerFAPL)
   do
      iter = iter + 1
      print('In the '..tostring(iter)..'th fastAPL iteration: the gap is '.. tostring(upperBound - lowerBound))
      print('The lower bound is '..tostring(lowerBound)..' and the upper bound is '..tostring(upperBound))
      hat_x, lowerBound, upperBound, proxCenter, Q = gapReduction(feval, fevalNOgrad, proxCenter, Q, lowerBound, upperBound, bundles, opts, radius)
      if fValueTrack.init then
	 fValueTrack.fTrack[1] = upperBound
	 fValueTrack.init = false
      else
	 fValueTrack.fTrack = torch.cat(fValueTrack.fTrack, torch.Tensor({upperBound}))
      end
      collectgarbage()
   end
   return hat_x, upperBound
end



function fastUSL.optimConstrained(feval, fevalNOgrad, opts, fValueTrack)
   -- setup specified parameters for unconstrained optimization
   local consOpt = {
      ballCenter = opts.ballConstraint.center,
      radius = opts.ballConstraint.radius,
      maxUnconsIter = opts.maxUnconsIter,
   }

   -- setup specified parameters for constrained bundle level optimization
   local bundleOpt = {
      ballCenter = opts.ballConstraint.center,
      beta = opts.beta,
      theta = opts.theta,
      initialLowerBound = opts.initialLowerBound,
      maxIterPerFAPL = opts.maxIterPerFAPL,
      maxIterPerFAPLgap = opts.maxIterPerFAPLgap,
      maxBundles = opts.maxBundles,
      maxPhases = opts.maxPhases,
      eps = opts.eps
   }
   
   local radius = opts.ballConstraint.radius
   local eta = 0
   local Q = 1
   local x_prime
   fastUSL.permuteTable = permuteMatrix.compute(opts.maxBundles + 2)
   -- initializations for fast APL algorithm
   local p0 = opts.ballConstraint.center:clone()
   local f_p0, grad_p0,f_p0_eta = feval(p0, eta)
   -- p1 \in \arg\min_x_{x \in BallCons(xbar,radius)} h(p0,x)
   -- where h(p0,x) is the linear approximation of feval at p0 with the form a*x + b
   -- so given a, within the centered ball, the min occurs at -a direction of unit ball surface
   local p1 = - radius * (grad_p0 / torch.norm(grad_p0))
   local f_p1, f_p1_eta = fevalNOgrad(p1, eta)
   local lowerBound = math.max(linearFunAtX(p0, f_p0, grad_p0)(p1), opts.initialLowerBound)
   -- local lowerBound = linearFunAtX(p0, f_p0, grad_p0)(p1)
   local upperBound
   if  (f_p0 < f_p1) then
      upperBound, x_prime =  f_p0, p0
   else
      upperBound, x_prime = f_p1, p1
   end
   -- compute the initial gap of Algorithm 1
   local proxCenter = x_prime
   x_prime = fastUSL.optim(feval, fevalNOgrad, bundleOpt, radius, Q, proxCenter, lowerBound, upperBound, fValueTrack)
   return x_prime
end

-- you should not use this function
-- function fastAPL.optimUnconstrained(feval, opts, fValueTrack)
--    -- setup specified parameters for unconstrained optimization
--    local unconsOpt = {
--       ballCenter = opts.ballConstraint.center:clone(),
--       radius = opts.ballConstraint.radius,
--       maxUnconsIter = opts.maxUnconsIter,
--       eps = opts.eps
--    }

--    -- setup specified parameters for constrained bundle level optimization
--    local bundleOpt = {
--       ballCenter = opts.ballConstraint.center:clone(),
--       beta = opts.beta,
--       theta = opts.theta,
--       initialLowerBound = opts.initialLowerBound,
--       maxIterPerFAPL = opts.maxIterPerFAPL,
--       maxIterPerFAPLgap = opts.maxIterPerFAPLgap,
--       maxBundles = opts.maxBundles,
--       maxPhases = opts.maxPhases,
--       eps = opts.eps
--    }
   
--    local radius = opts.ballConstraint.radius
--    local x_prime = opts.ballConstraint.center:clone()
--    local x_primeDouble = x_prime
--    -- initializations for fast APL algorithm
--    local p0 = opts.ballConstraint.center:clone()
--    local f_p0, grad_p0 = feval(p0)
--    -- p1 \in \arg\min_x_{x \in BallCons(xbar,radius)} h(p0,x)
--    -- where h(p0,x) is the linear approximation of feval at p0 with the form a*x + b
--    -- so given a, within the centered ball, the min occurs at -a direction of unit ball surface
--    local p1 = - radius * (grad_p0 / torch.norm(grad_p0))
--    local f_p1, grad_p1 = feval(p1)
   
--    local lowerBound = math.max(linearFunAtX(p0, f_p0, grad_p0)(p1), opts.initialLowerBound)
--    -- local lowerBound = linearFunAtX(p0, f_p0, grad_p0)(p1), opts.initialLowerBound)
--    local upperBound, upperBound_old
--    if  (f_p0 < f_p1) then
--       upperBound, x_prime =  f_p0, p0:clone()
--    else
--       upperBound, x_prime = f_p1, p1:clone()
--    end
   
--    -- compute the initial gap of Algorithm 1
--    local proxCenter = x_prime:clone()
   
--    local delta  = math.min(upperBound - lowerBound, opts.initialGap)
--    -- local delta  = upperBound - lowerBound
--    local deltaDecay = 1000
--    local iter = 0
--    local xmin
--    while delta > unconsOpt.eps and iter < unconsOpt.maxUnconsIter
--    do
--       iter = iter + 1
--       bundleOpt.eps = delta
--       repeat
-- 	 x_prime = fastAPL.optim(feval, bundleOpt, radius, proxCenter, lowerBound, upperBound, fValueTrack)
-- 	 x_primeDouble = fastAPL.optim(feval, bundleOpt, 2*radius, proxCenter, lowerBound, upperBound, fValueTrack, {init = true, fTrack = torch.Tensor(1)})
-- 	 if (feval(x_prime) - feval(x_primeDouble) > delta) then
-- 	    print('radius is expanding')
-- 	    radius = 2 * radius
-- 	 else
-- 	    break
-- 	 end
--       until (true)
      
--       delta = delta / 2
--    end

--    return x_primeDouble
-- end
