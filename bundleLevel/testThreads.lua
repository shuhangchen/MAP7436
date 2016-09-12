#!/Users/shuhang/torch/install/bin/th
local thread = require 'threads'
require 'torch'

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

local function mainTest()
   local size = 3
   return generateIndexMatrix(size)
end

local function testThread()
   local size = 3
   local M = torch.rand(size, size)
   local I = torch.eye(size)
   local total = torch.cat(M, I, 2)
   local lambda = torch.rand(size)
   local mu = torch.rand(size)
   local nthread = 5
   local njobs = torch.pow(2, size)
   local msg = "starting at this thread"
   local permuteMatrix = generateIndexMatrix(size)
   print(permuteMatrix)
   local permutePool = thread.Threads(
      nthread,
      function() end
   )

   local jobdone =  0
   for i=1, njobs do
      permutePool:addjob(
	 function()
	    return __threadid
	 end,
	 function(id)
	    local resMatrix = torch.Tensor(size, size)
	    resMatrix = total:maskedSelect(permuteMatrix[i]:eq(1):view(1, 2*size):expand(size, 2*size)):view(size, size)
	    print('permutaion matrix generated in thread'.. id)
	    print(resMatrix)
	 end
      )
   end

   permutePool:synchronize()
   print('jobs done')
   permutePool:terminate()
end

-- mainTest()

testThread()      
