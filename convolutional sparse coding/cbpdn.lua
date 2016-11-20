-- this module implements the ADMM algorithm for computing sparse codes given a dictionary

require 'torch'
local signal = require 'signal'
cbpdn = {}

local function shrink(v, lambda)
   return torch.cmul(torch.sign(v), torch.cmax(torch.abs(v) - lambda, 0))
end

local function complexConjN(x)
   -- take the conjugate of a multi-dimensional arrary dim1*dim2*dim3*2
   -- (not necessarily N*2 as in the signal processing package)
   local out2d = signal.complex.conj(x:view(-1, 2))
   return out2d:resizeAs(x)
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

local function fourierMul(x, y)
   -- the input sizes of x and y are supposed to be
   -- x-- dim1 * dim2 * dim3 * 2
   -- y-- dim1 * dim2 * dim3 * 2
   -- the complex.mm function requires the two variables have the size as dim1*dim2*2
   local out = torch.Tensor(x:size())
   for i = 1, x:size(3) do
      out[{{},{},i,{}}] = complexCmulN(x[{{},{},i,{}}], y[{{},{},i,{}}])
   end
   return out
end

local function solvedbi_sm(ah, rho, b)
   -- solve the diagonal block linear system with a scaled identity using the Sherman-Morrison equation
   -- (a a^H + rho I) x = b

   -- here, for now, just assuming ah and h takes the form of dim1*dim2*dim3*2
   -- dim1,dim2-- image row and column, dim3--number of filters, 2 is for complex tensor
   -- this function needs to be rewritten with respect to the complex tensor form
   -- and also compare the results with the matlab version to make sure it is right
   local a = complexConjN(ah)
   local c = torch.zeros(ah:size())
   local a3dsum = complexCmulN(ah, a):sum(3)
   a3dsum = a3dsum[{{},{},1,{1}}] + rho
   a3dsum = a3dsum:expand(a3dsum:size(1), a3dsum:size(2), 2)
   --because it is ah*a, so the img part of a3dsum should be zero
   -- right divide
   for i = 1, ah:size(3) do
      c[{{},{}, i,{}}] = torch.cdiv(ah[{{},{},i,{}}], a3dsum)
   end
   
   local cb = torch.zeros(ah:size(1), ah:size(2), 2)
   for i = 1, ah:size(3) do
      cb = cb + complexCmulN(c[{{},{},i,{}}], b[{{}, {}, i,{}}])
   end
   
   local cba = torch.zeros(ah:size(1), ah:size(2), ah:size(3), 2)
   for i = 1, ah:size(3) do
      cba[{{}, {}, {i},{}}] = complexCmulN(cb, a[{{},{},i,{}}])
   end
   return (b - cba) / rho
end

function cbpdn.solvedbi_sm(ah, rho, b)
   -- outter api for test purposes
   return solvedbi_sm(ah, rho, b)
end

local function fft2zeroPadding3D(x, m, n)
   local paddingx = torch.zeros(m, n, x:size(3))
   paddingx[{{1, x:size(1)}, {1, x:size(2)}, {}}] = x
   local out = torch.Tensor(m, n, x:size(3), 2) -- complex tensor in fourie domain
   for i= 1, x:size(3) do
      out[{{},{},i,{}}] = signal.fft2(paddingx[{{},{},i}])
   end
   return out
end

local function fft2for3D(x)
   local out = torch.zeros(x:size(1), x:size(2), x:size(3), 2)
   for i = 1, x:size(3) do
      out[{{},{},i,{}}] = signal.fft2(x[{{},{},i}])
   end
   return out
end

local function ifft2for3D(x)
   local out = torch.zeros(x:size(1), x:size(2), x:size(3))
   for i = 1, x:size(3) do
      out[{{},{},i}] = signal.ifft2(x[{{},{},i,{}}])[{{},{},1}]
   end
   return out
end

function cbpdn.admm(D, img, lambda, opts)
   -- the dictionary takes the form kerSize * kerSize * numFilters
   local DF = fft2zeroPadding3D(D, img:size(1), img:size(2))
   local numberOfFilters = D:size(3)
   local imgF = signal.fft2(img)
   local DimgF = torch.Tensor(imgF:size(1), imgF:size(2), numberOfFilters, 2):zero()
   local conjDF = complexConjN(DF)
   for i = 1, numberOfFilters do
      DimgF[{{},{},{i},{}}] = signal.complex.mm(conjDF[{{},{},i,{}}], imgF)
   end

   -- setup iteration parameters
   local rho = opts.rho
   if opts.rhoRsdlTarget == nil then
      opts.rhoRsdlTarget = 1 + math.pow(18.3, math.log10(lambda) + 1)
   end
   
   local r, s, rhomlt = 100, 100, 0
   local epri, edua = 0, 0
   local objValue = 0
   -- the vairbale spliting, x stands for sparse codes inside the data fitting term, and y is the splitted form of x
   local x = torch.zeros(img:size(1), img:size(2), numberOfFilters)
   local y = torch.zeros(img:size(1), img:size(2), numberOfFilters)
   local yPrev = y
   local u = torch.zeros(img:size(1), img:size(2), numberOfFilters)
   local iter = 0
   local objs = {}

   while iter < opts.maxIter and (r > epri or s > edua) 
   do
      -- solve the x subproblem
      -- the high memory solve thing?
      local xf = solvedbi_sm(DF, rho, DimgF + fft2for3D( y - u))
      x = ifft2for3D(xf)
      if opts.relax == 1 then
	 xr = x
      else
	 xr = opts.relax * x + (1 - opts.relax) * y
      end

      -- solve Y subproblem
      y = shrink( xr + u, (lambda/rho) * opts.l1weight)

      -- clear negative coefficient
      if opts.noNegCoef then
	 y[y:le(0)] = 0
      end

      if opts.noBndryCross then
	 y[{{-1 - D:size(1) + 2 ,- 1},{},{}}] = 0
	 y[{{},{-1 - D:size(2) + 2, -1}, {}}] = 0
      end 

      -- update the dual variable
      u = u + xr - y

      -- compute the objective function value in Fourier domain
      local dataFidelity = signal.complex.abs(fourierMul(DF, xf):sum(3):view(imgF:size(1), imgF:size(2), 2) - imgF):sum() / (2 * img:size(1) * img:size(2))
      objValue = dataFidelity + lambda * torch.abs(x * opts.l1weight):sum()

      -- compute the stopping criteria for admm
      local nx, ny, nu = torch.norm(x), torch.norm(y), torch.norm(u)
      r = torch.norm(x- y) / math.max(nx, ny)
      s = torch.norm(rho* ( y - yPrev))/ nu 
      epri = math.sqrt(nx) * opts.absStopTol/math.max(nx, ny) + opts.relStopTol
      edua = math.sqrt(nx) * opts.absStopTol/ (rho * nu) + opts.relStopTol

      -- update rho
      if opts.autoRho then
	 if iter ~= 1 and iter % opts.autoRhoPeriod == 0 then
	    if opts.autoRhoScaling then
	       rhomlt = math.sqrt(r / ( s * opts.rhoRsdlTarget))
	       if rhomlt < 1 then
		  rhomlt = 1 / rhomlt
	       end
	       if rhomlt > opts.rhoScaling then
		  rhomlt = opts.rhoScaling
	       end
	    else
	       rhomlt = opts.rhoScaling
	    end
	    local rsf = 1
	    if r > opts.rhoRsdlTarget * opts.rhoRsdlRatio * s then
	       rsf = rhomlt
	    end
	    if s > (opts.rhoRsdlRatio / opts.rhoRsdlTarget) * r then
	       rsf = 1/ rhomlt
	    end
	    rho = rsf * rho
	    u:div(rsf)
	 end
      end
      yPrev = y
      iter = iter + 1
      objs[iter] = objValue
      collectgarbage()
   end
   return y, objs
end   
	 
