require 'optim'
require 'nn'
require 'torch'

local CRF, parent = torch.class('nn.CRF', 'nn.Module')

function CRF:__init(NeighborMatirx) 
   parent.__init(self)
   self.NeighborMatirx = NeighborMatirx
   self.weight = torch.Tensor(1)   -- it is the log of beta before the neighboring
   self.bias = torch.Tensor(1)     -- it is the log of alpha for data fidelity term
   self.gradWeight = torch.Tensor(1)
   self.gradBias = torch.Tensor(1)
   self:reset() 
end

function CRF:reset(stdv) 
   -- initilize the weight and bias parameter
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))  -- slightly different than the way it is supposed to be (the size part )
   end

   self.weight[1] = torch.uniform(-stdv, stdv)
   self.bias[1] = torch.uniform(-stdv, stdv)
   
   return self
end


function CRF:updateOutput(input)
   -- we assume the updateOutput share the same input as the following updateGradInput and accGradParameters functions
   -- so we only calculate the matrice here once.
   self.B = torch.mul( input, 2 * torch.exp(self.bias[1]))
   self.D = torch.diag(torch.sum(self.NeighborMatirx, 2):resize(self.NeighborMatirx:size(1)))
   self.A = torch.mul(torch.eye(self.NeighborMatirx:size(1)), torch.exp(self.bias[1])) + torch.mul(self.D, torch.exp(self.weight[1])) - torch.mul(self.NeighborMatirx, torch.exp(self.weight[1])) 
   -- looks like we will have to use this inverse matrix in the following functions
   self.A_inv = torch.inverse(self.A)

   local feval = function(x) 
      collectgarbage() 
      if x:dim() ==  1 then 
	 -- this is a vector
	 local resEq = torch.mv(self.A, x) + torch.mul(self.B, 1/2)
	 local fun = torch.pow(torch.norm(resEq), 2)
	 local grad = torch.mv(torch.mul(self.A:t(),2), resEq)
	 return fun, grad 
      elseif x:dim() == 2 then
	 -- this is a batch, and the input is a matrix
	 local resEq = torch.mm(self.A, x:t()) + torch.mul(self.B:t(), 1/2)
	 local fun =torch.pow(torch.norm(resEq), 2)
	 local grad = torch.mm(torch.mul(self.A:t(),2), resEq)
	 return fun, grad
      else
	 error('input must be vector or matrix')
      end
   end
   -- we here use conjugate gradient to solve the linear equation As= -0.5b
   self.output = optim.cg(feval, input)
   -- Or we can use the inverse matrix directly
   -- if input:dim() == 1 then
   --    self.output = torch.mv(self.A_inv, self.B):mul(-0.5)
   -- elseif input:dim() == 2 then
   --    self.output = torch.mm(self.A_inv, self.B):mul(-0.5)
   -- end
   return self.output
end

function CRF:updateGradInput(input, gradOutput)
   -- the gradInput variable is initilized in the super class, 
   -- Module's init function (constructor)
   -- this updates the 

   -- this is the gradient for exp(alpha, beta) solve the problem !!!!!!!!!!!!11
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput.resizeAs(input)  -- resizeAs, has the same size, but value of entries is radnom if it has not been assigned value
      if self.gradInput:nElement() ~= nElement then  
	 -- why is this ? An: the gradinput variable is still in the way as it was initilized
	 -- so we need to set all its entries to zero
	 self.gradInput:zero()
      end
      -- mind the problem of synchronization
      -- !!!!!!!!!! RE-derive the equation, and mind the transpose issue.
      -- the 0 refers to beta in addmm, which means no accumulates
      if input:dim() == 1 then
	 self.gradInput:addmv(0, 1, torch.mul(self.A_inv, torch.exp(self.bias[1])), gradOutput)
      elseif input:dim() == 2 then
	 self.gradInput:addmm(0, 1 ,gradOutput, torch.mul(self.A_inv, torch.exp(self.bias[1])))
      end
      return self.gradInput
   end
end


function CRF:accGradParameters(input, gradOutput, scale)
   -- the addr, addmm, addmv !!!accumulate!!! the gradients
   scale = scale or 1
   if input:dim() == 1 then

      -- input is a vector
      local gradBias_inCRF = torch.add(torch.mv(torch.mm(self.A_inv, self.A_inv), torch.mul(self.B, -1)), 2, torch.mv(self.A_inv, input))
      self.gradGrad:addr(scale, gradOutput, torch.mul(gradBias_inCRF, -0.5* torch.exp(self.bias[1])))
      local gradWeight_inCRF = torch.mv(torch.mm(torch.mm(self.A_inv, torch.add(self.D, -1, self.NeighborMatirx)), self.A_inv), self.B)
      self.gradWeight:addr(scale, gradOutput, torch.mul(gradWeight_inCRF, 0.5))


   elseif input:dim() == 2 then
      -- input is a matrix
      -- mind the trandpose thing
      -- !!!!!! rederive the eqations
      local gradBias_inCRF = torch.add(torch.mm(torch.mm(self.A_inv, self.A_inv), torch.mul(self.B:t(), -1)), 2, torch.mm(self.A_inv, input:t()))
      self.gradGrad:addmm(scale, gradOutput:t(), torch.mul(gradBias_inCRF, -0.5* torch.exp(self.bias[1])))
      local gradWeight_inCRF = torch.mm(torch.mm(torch.mm(self.A_inv, torch.add(self.D, -1, self.NeighborMatirx)), self.A_inv), self.B:t())
      self.gradWeight:addmm(scale, gradOutput:t(), torch.mul(gradWeight_inCRF, 0.5))
   end
end

-- I don't understand this either
-- we do not need to accumulate parameters when sharing
CRF.sharedAccUpdateGradParameters = CRF.accUpdateGradParameters

function CRF:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.NeighborMatirx:size(2), self.NeighborMatirx:size(1))
end

   
   
