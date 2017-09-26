-- this file tests the hyperparameters for SGD method

-- roughly speaking, it randomly sample different settings for SGD model and evaluate the resulting model

-- only need to change the nesting structure of the for loop and parameter settings


-- only random  nested grid searching
-- set parameters for hyper-experiments

require 'trainCond'
require 'torch'
require 'hypero'
require 'cutorch'
-- torch.manualSeed()
-- hyper parameters settings for training factor model
cutorch.setDevice(1)
hopt = {              
   batchSize = 100,
   maxNumEpoch = 150,
   LR = 4e-5,   --2e-5 ?   
   optimMethod = 'ADAM',             --probabilities for selecting those values  'SGD','ADAM','RMSprop'
   numHiddenUnits = 1500,  -- { 2, 3, 3, 2},     --probabilities for selecting those values 800, 1000, 1200, 1500 
   numHiddenUnits_VT = 1000,
   marginVT =torch.Tensor({6,8}),
   hingeCoef ={0.005, 0.01},
   noiseType = {'Gaussian','masking'}, 
   GaussianCorruptionVar = {0.01,0.5,1},
   maskCorruptionVar ={0.1,0.25,0.4} ,  -- 2.5,2.5,2.5,2.5 {0.1, 0.15, 0.25, 0.4}
   momentum = 0.9,
   weightDecay = 0.0001,
   sampleEachEpoch = 20000,          -- 10000, 20000, 40000
   RMSalpha =0.9,
   coefL1 = 0,                   -- L1 and L2 penalty coefficient for weight matrix
   coefL2 = 0.0001,
   cuda = true
   }

-- no evaluations


print('<Hyper optimization for '.. hopt.optimMethod..' has begin.\n')
for j=1,hopt.marginVT:size(1) do
   for k=1, #hopt.hingeCoef do
      for nt=1,2 do -- #hopt.noiseType do
	 for nc =1,#hopt.maskCorruptionVar do
	    
	    local opt = _.clone(hopt)

	    -- set hyper parameters for training factor model
	    local hp = {}
	    hp.batchSize =  opt.batchSize
	    hp.learningRate =  opt.LR
	    hp.optimMethod = opt.optimMethod
	    hp.numHiddenUnits = opt.numHiddenUnits
	    hp.numHiddenUnits_VT = opt.numHiddenUnits_VT
	    hp.numHiddenUnits_PA = opt.numHiddenUnits - opt.numHiddenUnits_VT
	    hp.marginVT = opt.marginVT[j]
	    hp.hingeCoef = opt.hingeCoef[k]
	    -- only one type of noise can be used
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
	    hp.cuda = opt.cuda
	    
	    -- -- train the factor model with above hyper-parameters
	    trainCond.train(hp,j,k,nt,nc)
	    
	    collectgarbage()
	 end
      end
   end
end


print('<Hyper optimization for conditional factor has finished.\n')
      
   
