local thread = require 'threads'
require 'torch' 

dualSolver = {}

dualSolver.numThreads = 8

local function permuteIndex(lengthDualVar)
   -- 1 stands for that dual variable is not zero, 0 is the dual of the dual is not zero
   if (lengthDualVar == 1) then
      return {{1},{0}}
   else
      local index = {}
      local indexOfLast = permuteIndex(lengthDualVar - 1)
      for i,v in ipairs(indexOfLast) do
	 local subindex1 = {}
	 local subindex0 = {}
	 for j,q in ipairs(v) do
	    table.insert(subindex1, q)
	    table.insert(subindex0, q)
	 end
	 table.insert(subindex1, 1)
	 table.insert(subindex0, 0)
	 index[i] = subindex0
	 index[i + #indexOfLast] = subindex1
      end
      return index
   end
end

local function generateIndexMatrix(lengthDualVar)
   local index = permuteIndex(lengthDualVar)
   local matrixOfpermutation = torch.Tensor(torch.pow(2, lengthDualVar), 2 * lengthDualVar)
   assert(#index == matrixOfpermutation:size(1), " the generation of column table index is wrong")
   -- print(columnIndexTable)
   for i,v in ipairs(index) do
      local vector = torch.Tensor(v)
      vector = torch.cat(vector, 1 - vector)
      -- print(vector)
      matrixOfpermutation[i] = vector:clone()
   end
   return matrixOfpermutation
end

function dualSolver.solve(bigM, c)
   --- it returns two variables
   -- the first one is boolean, indicates whether it has found the legitimate solution or not
   -- the second variable is the lambdaAndMu tensor, it is all zero if first returned value is false
   assert(bigM:size(1) * 2 == bigM:size(2), 'Wrong shape of the big M')
   local size = bigM:size(1)
   local lambdaAndMu = torch.Tensor(bigM:size(2)):zero()
   local nthread = dualSolver.numThreads
   local njobs = torch.pow(2, bigM:size(1))
   local permuteMatrix = generateIndexMatrix(bigM:size(1))
   -- print(permuteMatrix:size())
   local solutionPool = thread.Threads(
      nthread,
      function() end
   )
   local solutionFound = false
   for i=1, njobs do
      if not solutionFound then
	 solutionPool:addjob(
	    function()
	       local resMatrix = torch.Tensor(size, size)
	       local resVec = torch.Tensor(size)
	       resMatrix = bigM:maskedSelect(permuteMatrix[i]:eq(1):view(1, 2 * size):expand(size, 2*size)):view(size, size)
	       local tempSolution
	       local isFullSolution = false
	       -- define the function
	       -- smallM*smallLambdaMu = c
	       local solveLinearEquation= function() tempSolution = torch.gesv(c:view(-1, 1), resMatrix) return tempSolution end
	       if pcall(solveLinearEquation) then
		  -- this equation has a solution because 
		  tempSolution = torch.gesv(c:view(-1,1), resMatrix)
		  isFullSolution = true
	       else
		  -- -- the linear equation smallM*smallLambdaMu = c does not have a solution, smallM is singular
		  -- -- in this case, we consider the following problem instead
		  -- -- we consider the following least squares problems for a ATA = c. ||AX-B||_F.
		  -- local solveLinearEquationByLS = function() tempSolution = torch.gels(torch.mv(resMatrix:t(), c):view(-1,1), resMatrix:t() * resMatrix) return tempSolution end
		  -- -- if it still has error, we just return nil, this equation has no solution then.
		  -- if pcall(solveLinearEquationByLS) then
		  --    -- the alternative equation has a solution
		  --    tempSolution = solveLinearEquationByLS()
		  --    isFullSolution = false
		  -- else
		  --    return nil, false
		  -- end
		  return nil, false
	       end
	       if tempSolution:min() >= 0 then
		  -- it is legitimate solution, we need to check the distance between MX, b
		  if torch.dist(resMatrix * tempSolution, c) < 1e-12 then
		     -- we have find a solution, time to abort this task and end all other threads
		     if isFullSolution then
			return tempSolution,true
		     end
		  end
	       end
	       return nil
	    end,
	    function(tempSolution, isFullSolution)
	       if tempSolution then
		  -- we have found the solution
		  if isFullSolution then
		     solutionFound = true
		     lambdaAndMu:zero()
		     lambdaAndMu:maskedCopy(permuteMatrix[i]:eq(1), tempSolution)
		  end
	       end
	    end
	 )
      else
	 break
      end
   end

   solutionPool:synchronize()
   solutionPool:terminate()
   
   return solutionFound, lambdaAndMu
end
