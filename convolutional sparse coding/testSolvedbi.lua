require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'cbpdn'
require 'image'
local signal = require 'signal'
local matio = require 'matio'

local function fft2zeroPadding3D(x, m, n)
   local paddingx = torch.zeros(m, n, x:size(3))
   paddingx[{{1, x:size(1)}, {1, x:size(2)}, {}}] = x
   local out = torch.Tensor(m, n, x:size(3), 2) -- complex tensor in fourie domain
   for i= 1, x:size(3) do
      out[{{},{},i,{}}] = signal.fft2(paddingx[{{},{},i}])
   end
   return out
end

local function complexConjN(x)
   -- take the conjugate of a multi-dimensional arrary dim1*dim2*dim3*2
   -- (not necessarily N*2 as in the signal processing package)
   local out2d = signal.complex.conj(x:view(-1, 2))
   return out2d:resizeAs(x)
end

local function fft2for3D(x)
   local out = torch.zeros(x:size(1), x:size(2), x:size(3), 2)
   for i = 1, x:size(3) do
      out[{{},{},i,{}}] = signal.fft2(x[{{},{},i}])
   end
   return out
end
local function complexCmulN(x, y)
   -- compute the element-wise multiplication of two 3-d array, the forth dimension is 2, (the complex)
   if x:isContiguous() and y:isContiguous() then
      local out2d = signal.complex.cmul(x:view(-1, 2), y:view(-1, 2))
      return out2d:viewAs(x)
   else
      local out2d = torch.Tensor(x:size())
      for i = 1, out2d:size(2) do
	 out2d[{{},i,{}}] = signal.complex.cmul(x[{{},i,{}}], y[{{},i,{}}])
      end
      return out2d
   end
end


local img = image.load('lena.png', 1):div(255)
img = img:sub(1, 30, 1, 30)
local imgF = signal.fft2(img)
local rho = 1
local dic = matio.load('ConvDict.mat').ConvDict.Dict[5]
local numberOfFilters = dic:size(3)
local DF = fft2zeroPadding3D(dic, img:size(1), img:size(2))
local DimgF = torch.Tensor(imgF:size(1), imgF:size(2), numberOfFilters, 2):zero()
local y = torch.zeros(img:size(1), img:size(2), numberOfFilters)
local u = torch.zeros(img:size(1), img:size(2), numberOfFilters)
local conjDF = complexConjN(DF)
for i = 1, numberOfFilters do
   DimgF[{{},{},{i},{}}] = complexCmulN(conjDF[{{},{},i,{}}], imgF)
end
local solvec = DimgF + fft2for3D(y - u)
local xf = cbpdn.solvedbi_sm(DF, rho, DimgF + fft2for3D(y - u))			   

matio.save('solve.mat',{img = img, imgF = imgF, dic = dic, DimgF = DimgF, conjDF = conjDF, DF = DF, rho = rho, solvec = solvec, xf = xf})
