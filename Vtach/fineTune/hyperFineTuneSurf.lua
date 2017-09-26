
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'cunn'
require 'fineTuneSurf'
require 'hypero'
require 'cutorch'
cutorch.setDevice(1)
hopt = {
   MaxNumEpoches = 700, ----700
   optimization = 'SGD', -- 'LBFGS',
   learningRate ={ 2e-3},    -- 5E-3 FOR sgd
   momentum = 0.9,   -- 0.95
   RMSalpha = 0.9,   -- RMSprop only
   maxIter = 5,
   weightDecay = 0.0001,
   coefL1 = 0.01,                   -- L1 and L2 penalty coefficient for weight matrix
   coefL2 = 0,
   dropout = true,
   batchNormalization=false,
   surfaceCri= true,
   project = false,
   surfParam = {5},
   cuda = true
}


for j = 1, #hopt.learningRate do
   for k = 1, #hopt.surfParam do
      local opt = _.clone(hopt)
      -- set hyper parameters for training factor model
      local hp = {}
      hp.learningRate =  opt.learningRate[j]
      hp.optimization = opt.optimization
      hp.momentum = opt.momentum
      hp.weightDecay = opt.weightDecay
      hp.maxIter = opt.maxIter
      hp.RMSalpha = opt.RMSalpha
      hp.coefL1 = hp.coefL1
      hp.coefL2 = hp.weightDecay
      hp.MaxNumEpoches = opt.MaxNumEpoches
      hp.cuda = opt.cuda
      hp.dropout = opt.dropout
      hp.batchNormalization = opt.batchNormalization
      hp.surfaceCri = opt.surfaceCri
      hp.project = opt.project
      hp.surfParam = opt.surfParam[k]
      -- -- train the factor model with above hyper-parameters
      print('<Hyper optimization for MSE '..tostring(j)..' '..tostring(k))
      fineTuneSurf.run(hp,j,k)
      collectgarbage()
   end
end


      
