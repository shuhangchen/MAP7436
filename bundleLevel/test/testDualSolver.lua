require 'dualSolver'

local size = 9
local M = torch.randn(size, size)
M = M * M:t()
local bigM = torch.cat(M, - torch.eye(size), 2)
local smallM = torch.Tensor(size, size)
local x = torch.rand(size * 2)
local randSelect = x[{{1,size}}]:ge(0.5)
local colSelector = torch.Tensor(2 * size):zero()

-- select cols
for i=1,size do
   if randSelect[i] == 1 then
      -- lambda[i] is not zero so mu[i] has to be equal
      colSelector[i] = 1
      x[i + size] = 0
   else
      x[i] = 0
      colSelector[i+size] = 1
   end
end
local selectedCol = 0
for i = 1, 2* size do
   if colSelector[i] == 1 then
      selectedCol = selectedCol + 1
      smallM[{{},{selectedCol}}] = bigM[{{}, {i}}]
   end
end
local c = torch.mv(bigM, x)
print(bigM)
print(smallM)
print('----- solving now -----')

local solutionFound, dualSolution = dualSolver.solve(bigM,c)

-- checking ansers
print(x)
print(dualSolution)
if solutionFound then
   print(torch.norm(x - dualSolution))
   print(torch.norm(bigM * dualSolution - c))
else
   if (pcall(torch.inverse(smallM))) then
      print('no solution but it should have')
   else
      print('no solution but it is non-invertible '..tostring(torch.norm(x-dualSolution)))
   end
end

