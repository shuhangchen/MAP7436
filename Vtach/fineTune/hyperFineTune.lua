
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'cunn'
require 'fineTune'
require 'hypero'
require 'cutorch'
cutorch.setDevice(2)

hopt = {
   MaxNumEpoches = 700, ----700
   optimization = {'SGD'}, -- 'LBFGS',
   learningRate ={3e-3},    -- 5E-3 FOR sgd
   momentum = {0.9},   -- 0.95
   RMSalpha = 0.9,   -- RMSprop only
   maxIter = 5,
   weightDecay = {0.0001},
   coefL1 = 0.01,                   -- L1 and L2 penalty coefficient for weight matrix
   coefL2 = 0,
   dropout = true,
   batchNormalization=false,
   surfaceCri= false,
   cuda = true
}


for j = 1, #hopt.optimization do
for k=1, #hopt.learningRate do
      for wd = 1, #hopt.weightDecay do
	 local opt = _.clone(hopt)
	 print('<Hyper optimization for MSE '..tostring(j)..' '..tostring(k)..' '..tostring(wd))
	 -- set hyper parameters for training factor model
	 local hp = {}
	 hp.learningRate =  opt.learningRate[k]
	 hp.optimization = opt.optimization[j]
	 hp.momentum = opt.momentum
	 hp.weightDecay = opt.weightDecay[wd]
	 hp.maxIter = opt.maxIter
	 hp.RMSalpha = opt.RMSalpha
	 hp.coefL1 = hp.coefL1
	 hp.coefL2 = hp.weightDecay
	 hp.MaxNumEpoches = opt.MaxNumEpoches
	 hp.cuda = opt.cuda
	 hp.dropout = opt.dropout
	 hp.batchNormalization = opt.batchNormalization
	 hp.surfaceCri = opt.surfaceCri
	 -- -- train the factor model with above hyper-parameters
	 fineTune.run(hp, j, k, wd)
	 collectgarbage()
      end
   end
end


      
