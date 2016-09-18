require 'torch'

permuteMatrix = {}

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

function permuteMatrix.compute(size)
   permuteMatrices = {}
   for i = 2, size do
      table.insert(permuteMatrices, generateIndexMatrix(i))
   end
   return permuteMatrices
end
