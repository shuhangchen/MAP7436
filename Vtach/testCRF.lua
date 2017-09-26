require 'CRF'
require 'nn'
require 'torch'
require 'optim'

-- we first test the single CRF module and then we test it within the sequential modul
local input = torch.rand(2,4)

local neighboring = torch.Tensor({{1,1,0 , 0}, {1, 0, 1,0}, {0,1,1,1}, {0,0,1,1}}) 

local crf = nn.CRF(neighboring)

local output = crf:forward(input)
