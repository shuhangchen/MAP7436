-- this file tests the hyperparameters for SGD method

-- roughly speaking, it randomly sample different settings for SGD model and evaluate the resulting model

-- only need to change the nesting structure of the for loop and parameter settings


-- only random  nested grid searching
-- set parameters for hyper-experiments

require 'condFactor'
require 'torch'
require 'hypero'
require 'cutorch'

cutorch.setDevice(2)
-- torch.manualSeed()
-- hyper parameters settings for training factor model
hopt = {              
   batchSize = 100,
   maxNumEpoch = 150,
   LR = 1e-5,    
   optimMethod = 'ADAM',             --probabilities for selecting those values  'SGD','ADAM','RMSprop'
   numHiddenUnits = {500, 600},  -- { 2, 3, 3, 2},     --probabilities for selecting those values 800, 1000, 1200, 1500
   --  warpped  
   marginMain =torch.Tensor({5,6}),
   hingeCoef ={0.1, 0.5, 0.8},
   noiseType = {'Gaussian','masking'}, 
   GaussianCorruptionVar = {0.00001, 0.0001,0.01, 0.05, 0.1},
   maskCorruptionVar ={0.05,0.1,0.15,0.25, 0.4},
   -- 2.5,2.5,2.5,2.5 {0.1, 0.15, 0.25, 0.4}
   momentum = 0.9,
   weightDecay = 0.0001,
   sampleEachEpoch = 20000,          -- 10000, 20000, 40000
   RMSalpha =0.9,
   coefL1 = 0,                   -- L1 and L2 penalty coefficient for weight matrix
   coefL2 = 0.0001,
   trainVT= true,
   cuda = true
   }

-- no evaluations



print('<Hyper optimization for conditioanl auto-encoder has begin.\n')
for i = 2, #hopt.numHiddenUnits do
   for j=1,hopt.marginMain:size(1) do
      for k=1, #hopt.hingeCoef do
	 for nt=1,1 do --#hopt.noiseType do  -- only do gaussian noise
	    for nc =1,#hopt.maskCorruptionVar do
	       
	       local opt = _.clone(hopt)

	       -- set hyper parameters for training factor model
	       local hp = {}

	       hp.batchSize =  opt.batchSize
	       hp.learningRate =  opt.LR
	       hp.optimMethod = opt.optimMethod
	       hp.numHiddenUnits = opt.numHiddenUnits[i] 
	       hp.marginMain = opt.marginMain[j]
	       hp.hingeCoef = opt.hingeCoef[k]
	       hp.noiseType = opt.noiseType[nt]
	       if hp.noiseType == 'Gaussian' then
		  hp.GaussianCorruptionVar = opt.GaussianCorruptionVar[nc]
		  hp.maskCorruptionVar = 0
	       else
		  hp.maskCorruptionVar = opt.maskCorruptionVar[nc]
		  hp.GaussianCorruptionVar = 0
	       end
	       hp.momentum = opt.momentum
	       hp.weightDecay = opt.weightDecay
	       hp.sampleEachEpoch = opt.sampleEachEpoch
	       hp.maxNumEpoch = opt.maxNumEpoch
	       hp.maxIter = opt.maxIter
	       hp.RMSalpha = opt.RMSalpha
	       hp.coefL1 = opt.coefL1
	       hp.coefL2 = opt.coefL2
	       hp.trainVT = opt.trainVT
	       hp.cuda = opt.cuda
	       
	       -- -- train the factor model with above hyper-parameters
	       condFactor.train(hp,i,j,k,nt,nc)
	       
	       collectgarbage()
	    end
	 end
      end
   end
end
print('<Hyper optimization for conditional factor has finished.\n')
      
   
