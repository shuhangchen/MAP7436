
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'

condFactor={}

function condFactor.train(opt,i,j,k,nt,nc)

   -- opt = {
   --    learningRate = 2e-5,
   --    weightDecay = 0.0001,
   --    momentum = 0.9,
   --    sampleEachEpoch = 20000,
   --    marginMain = 9,
   --    hingeCoef = 0.02,
   --    GaussianCorruptionVar = 0,
   --    maskCorruptionVar = 0.15,         --0.3
   --    maxNumEpoch = 150,
   --    batchSize = 100,
   --    numHiddenUnits = 1200,
   --    optimMethod = 'ADAM',
   --    coefL1 = 0,                   -- L1 and L2 penalty coefficient for weight matrix
   --    coefL2 = 0.0001,
   --    trainVT= true,
   --    cuda = true
   -- }
   -- close all gnuplots
   os.execute('killall gnuplot')


   -- enter the default director to save results
   hexperName = '/home/mark/Documents/VT_learning/Logs/factor3layer/MES_i'..tostring(i)..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
   os.execute('mkdir -p '.. hexperName)
   torch.save(hexperName..'/hypers.t7',opt)
   -- you can only choose one type of noise 
   assert( ((opt.GaussianCorruptionVar and opt.maskCorruptionVar)  and (opt.GaussianCorruptionVar or opt.maskCorruptionVar)), " the denoising parameter is not set propoerly")




   -- load the QRS data
   qrsData_loaded = torch.load('/home/mark/Documents/VT_learning/squarePA/secondLayer/candidates/qrsLayer2.t7')

   -- load the QRS pair data
   qrsPairs_loaded = torch.load('/home/mark/Documents/VT_learning/data/qrsPairs_coord.t7')
   if opt.cuda then
      --   qrsData.train_x = qrsData.train_x:cuda()
      qrsPairs_loaded = qrsPairs_loaded:cuda()       -- -- TODO: he file format is (each row) [index_example1 index_example2 labelVT(-1 or 1), label patient(-1 or 1)]
   end
   --use detailed clone
   qrsData = {
      train_x = {qrsData_loaded.train_x[1]:clone(),qrsData_loaded.train_x[2]:clone()},
      train_y = qrsData_loaded.train_y:clone(),
      val_x = {qrsData_loaded.val_x[1]:clone(), qrsData_loaded.val_x[2]:clone()},
      val_y = qrsData_loaded.val_y:clone(),
      train_coord = qrsData_loaded.train_coord:clone(),
      val_coord = qrsData_loaded.val_coord:clone()
   }
   qrsPairs = qrsPairs_loaded:clone()
   opt.trainVT = true
   if not opt.trainVT then
      -- we are training the conditional factor model for VT nodes
      qrsData.train_x[1],qrsData.train_x[2] = qrsData_loaded.train_x[2], qrsData_loaded.train_x[1]
      qrsPairs[{{},{3}}],qrsPairs[{{},{4}}] = qrsPairs_loaded[{{},{4}}],qrsPairs_loaded[{{},{3}}]
   end

   dataNum = qrsData.train_x[1]:size(1)
   dataDimMain = qrsData.train_x[1]:size(2) 
   dataDimAuxi = qrsData.train_x[2]:size(2)
   dataDim = dataDimMain + dataDimAuxi
   pairNum = qrsPairs:size(1)

   -- setup hidden unit architecture
   numHiddenUnits = opt.numHiddenUnits

   batchSize = opt.batchSize
   opt.sampleEachEpoch = opt.sampleEachEpoch - opt.sampleEachEpoch% batchSize  -- throw the lag data alway


   -- construct the netwrok architecture  
   -- encoder part
   encoder_example1 = nn.Sequential()
   encoder_example1:add(nn.JoinTable(2))
   encoder_example1:add(nn.Linear(dataDim,numHiddenUnits))
   encoder_example1:add(nn.Sigmoid())

   encoder_example2 = nn.Sequential()
   encoder_example2:add(nn.JoinTable(2))
   encoder_example2:add(nn.Linear(dataDim,numHiddenUnits))
   encoder_example2:add(nn.Sigmoid())

   -- two example tables should share weights and bias
   encoder_example2:get(2).weight:set(encoder_example1:get(2).weight)
   encoder_example2:get(2).bias:set(encoder_example1:get(2).bias)

   -- !!! pay attenion here, I am using optim package (flattened paramters), so I have to share gradients here, But I would need to do this if I use the updateGradParameters in my training procedure
   encoder_example2:get(2).gradWeight:set(encoder_example1:get(2).gradWeight)
   encoder_example2:get(2).gradBias:set(encoder_example1:get(2).gradBias)

   encoder_eg1andeg2 = nn.ParallelTable()
   encoder_eg1andeg2:add(encoder_example1)
   encoder_eg1andeg2:add(encoder_example2)

   encoder = nn.Sequential()
   encoder:add(encoder_eg1andeg2)


   -- decoder network
   decoder_example1 = nn.Sequential()
   decoder_example1:add(nn.Linear(numHiddenUnits,dataDimMain))
   decoder_example1:add(nn.Sigmoid())

   decoder_example2 = nn.Sequential()
   decoder_example2:add(nn.Linear(numHiddenUnits,dataDimMain))
   decoder_example2:add(nn.Sigmoid())


   -- make sure that the decoder example networks share weights
   decoder_example2:get(1).weight:set(decoder_example1:get(1).weight)
   decoder_example2:get(1).bias:set(decoder_example1:get(1).bias)

   -- !!! pay attenion here, I am using optim package (flattened paramters), so I have to share gradients here, But I would need to do this if I use the updateGradParameters in my training procedure
   decoder_example2:get(1).gradWeight:set(decoder_example1:get(1).gradWeight)
   decoder_example2:get(1).gradBias:set(decoder_example1:get(1).gradBias)

   decoder = nn.ParallelTable()
   decoder:add(decoder_example1)
   decoder:add(decoder_example2)

   -- concatenate the encoder and decoder into the factor model
   factor = nn.Sequential()
   factor:add(encoder)
   factor:add(decoder)


   -- define the loss function for the denoising reconstruction part
   p1_criterion = nn.BCECriterion() 
   p2_criterion = nn.BCECriterion() 
   denoising_criterion = nn.ParallelCriterion():add(p1_criterion,0.5):add(p2_criterion,0.5)

   if opt.cuda then
      factor:cuda()
      denoising_criterion:cuda()
   end
   -- command line prompt
   --   print('Training two-way factored denoising autoencoder\n'.. factor:__tostring());


   -- retrieve parameters and gradients (flattened)
   parameters, gradients = factor:getParameters()

   --pulls out the memory location staff out of the for loop for computing efficiency
   local batchx_eg1 = {torch.Tensor(batchSize,dataDimMain),torch.Tensor(batchSize,dataDimAuxi)}
   local batchy_eg1 = {torch.Tensor(batchSize,dataDimMain),torch.Tensor(batchSize,dataDimAuxi)}
   local batchx_eg2 = {torch.Tensor(batchSize,dataDimMain),torch.Tensor(batchSize,dataDimAuxi)}
   local batchy_eg2 = {torch.Tensor(batchSize,dataDimMain),torch.Tensor(batchSize,dataDimAuxi)}
   local VTPAlabels = torch.Tensor(batchSize,2)
   -- mallocate memory here
   local mlp_main = nn.PairwiseDistance(2)
   -- transform into cuda() version
   local hingeMain = nn.HingeEmbeddingCriterion(opt.marginMain)
   -- transform into cuda() version
   if opt.cuda then
      mlp_main:cuda()
      hingeMain:cuda()
      batchx_eg1[1] = batchx_eg1[1]:cuda()
      batchx_eg1[2] = batchx_eg1[2]:cuda()
      batchy_eg1[1] = batchy_eg1[1]:cuda()
      batchy_eg1[2] = batchy_eg1[2]:cuda()
      batchx_eg2[1] = batchx_eg2[1]:cuda()
      batchx_eg2[2] = batchx_eg2[2]:cuda()
      batchy_eg2[1] = batchy_eg2[1]:cuda()
      batchy_eg2[2] = batchy_eg2[2]:cuda()
      VTPAlabels = VTPAlabels:cuda()
   end
   if (opt.optimMethod == 'SGD') then   -- don't repeat weight decay for SGD method
      opt.coefL2 = 0
   end
   gradHinge = {}



   -- start training
   epoch=0

   while epoch < opt.maxNumEpoch do
      xlua.progress(epoch, opt.maxNumEpoch)
      epoch = epoch +1

      local lossFun = 0

      local time = sys.clock();
      local shuffle = torch.randperm(pairNum)[{{1,opt.sampleEachEpoch}}]
      
      -- training one epoch
      for i= 1 ,opt.sampleEachEpoch , batchSize do
	 -- display progress
	 --xlua.progress(i+batchSize-1,dataNum)
	 
	 local k=1
	 
	 for j = i, i+batchSize-1 do
	    batchy_eg1[1][k] = qrsData.train_x[1][qrsPairs[shuffle[j]][1]]:clone()
	    batchy_eg1[2][k] = qrsData.train_x[2][qrsPairs[shuffle[j]][1]]:clone()
	    batchy_eg2[1][k] = qrsData.train_x[1][qrsPairs[shuffle[j]][2]]:clone()
	    batchy_eg2[2][k] = qrsData.train_x[2][qrsPairs[shuffle[j]][2]]:clone()
	    VTPAlabels[k] = qrsPairs[{{shuffle[j]},{3,4}}]:clone()
	    k = k +1
	 end
	 -- add noise to corrupt the input
	 if (opt.GaussianCorruptionVar) then
	    -- Gaussian additive noise
	    if opt.cuda  then
	       batchx_eg1[1] = batchy_eg1[1] +  torch.randn(batchSize,dataDimMain):cuda():mul(opt.GaussianCorruptionVar)	   
	       batchx_eg2[1] = batchy_eg2[1] +  torch.randn(batchSize,dataDimMain):cuda():mul(opt.GaussianCorruptionVar)	   
	    else
	       batchx_eg1[1] = batchy_eg1[1] +  torch.randn(batchSize,dataDimMain):mul(opt.GaussianCorruptionVar)
	       batchx_eg2[1] = batchy_eg2[1] +  torch.randn(batchSize,dataDimMain):mul(opt.GaussianCorruptionVar)
	    end
	 else
	    -- masking noise
	    if opt.cuda then
	       batchx_eg1[1] = torch.cmul(batchy_eg1[1],torch.ge(torch.rand(batchSize,dataDimMain),opt.maskCorruption):double():cuda())
	       batchx_eg2[1] = torch.cmul(batchy_eg2[1],torch.ge(torch.rand(batchSize,dataDimMain),opt.maskCorruption):double():cuda())
	    else
	       batchx_eg1[1] = torch.cmul(batchy_eg1[1],torch.ge(torch.rand(batchSize,dataDimMain),opt.maskCorruption):double())
	       batchx_eg2[1] = torch.cmul(batchy_eg2[1],torch.ge(torch.rand(batchSize,dataDimMain),opt.maskCorruption):double()) 
	    end
	 end
	 batchx_eg1[2] = batchy_eg1[2]:clone()
	 batchx_eg2[2] = batchy_eg2[2]:clone()

	 local feval = function(x)
	    
	    collectgarbage()

	    if x ~= parameters then
	       parameters:copy(x)
	    end
	    
	    -- reset parameters
	    factor:zeroGradParameters()
	    -- backpropogate gradients for whole factor
	    local pred = factor:forward({batchx_eg1, batchx_eg2})
	    local errRecons = denoising_criterion:forward(pred,{batchy_eg1[1],batchy_eg2[1]})
	    local dw = denoising_criterion:backward(pred, {batchy_eg1[1],batchy_eg2[1]})
	    factor:backward({batchx_eg1, batchx_eg2}, dw)
	    

	    local function VTPACriterion(input, VTPAlabels)
	       -- 0.5* (pairDistance_VT + pairDistance_PA)
	       -- define the mlp_vt and pa model out side the loop
	       local distanceMain = mlp_main:forward({input[1], input[2]})

	       local errHingeMain = hingeMain:forward(distanceMain, VTPAlabels[{{},{1}}])

	       -- multiple the coefficient to loss function
	       errHinge = 0.5*opt.hingeCoef*errHingeMain
	       -- backpropagate
	       local gradHingeMain = hingeMain:backward(distanceMain, VTPAlabels[{{},{1}}])
	       mlp_main:zeroGradParameters()
	       local gradOutputMain = mlp_main:backward({input[1], input[2]}, gradHingeMain)
	       
	       -- multiply the coefficient into gradient vector
	       gradHinge[1] =  gradOutputMain[1]:mul(0.5*opt.hingeCoef)
	       gradHinge[2] =  gradOutputMain[2]:mul(0.5*opt.hingeCoef)

	       return errHinge, gradHinge
	    end
	    -- backpropagate gradients for encoder of pairwise distance
	    lossHinge, gradHinge = VTPACriterion(factor:get(1).output,VTPAlabels)
	    encoder:backward({batchx_eg1,batchx_eg2},gradHinge)
	    
	    local err = errRecons + lossHinge
	    --add penalty function for weight matrix
	    if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
	       
	       local norm, sign = torch.norm,torch.sign
	       
	       err = err + opt.coefL1* norm(parameters,1)
	       err = err + opt.coefL2* norm(parameters,2)^2/2
	       -- update gradients accordingly
	       gradients:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
	       
	    end
	    return err, gradients
	 end
	 
	 -- optimize on current mini-batch
	 if opt. optimMethod == 'ADAM' then

	    -- Perform ADAM step:
	    local adamConfig = adamConfig or {
	       learningRate = opt.learningRate
					     }
	    local adamState = adamState or {}
	    x,batchlossFun= optim.adam(feval, parameters, adamConfig,adamState)
	    

	 elseif opt. optimMethod == 'SGD' then
	    -- Perform SGD step:
	    local sgdState = sgdState or {
	       learningRate = opt.learningRate,
	       momentum = opt.momentum,
	       weightDecay = opt.weightDecay,
	       learningRateDecay = 5e-7
					 }
	    x,batchlossFun = optim.sgd(feval, parameters, sgdState)
	 elseif opt. optimMethod == 'RMSprop' then
	    local rmspropConfig = rmspropConfig or {
	       learningRate = opt.learningRate,
	       alpha = opt.RMSalpha
						   }
	    local rmsState = rmsState or {}
	    x, batchlossFun = optim.rmsprop(feval, parameters, rmspropConfig)
	 else
	    error('unknown optimization method')
	 end
	 
	 lossFun = lossFun + batchlossFun[1]
      end
      print("\nEpoch: " .. epoch .. " loss function: " .. lossFun/opt.sampleEachEpoch .. " time: " .. sys.clock() - time)

      if epoch >1 then
	 lossFunTrack = torch.cat(lossFunTrack, torch.Tensor(1,1):fill(lossFun/opt.sampleEachEpoch*batchSize),1)
      else
	 lossFunTrack = torch.Tensor(1,1):fill(lossFun/opt.sampleEachEpoch*batchSize)
      end
      

   end

   -- save the model
   -- print('Saving the model and loss function track to files')
   torch.save(hexperName..'/cond.net', factor)
   torch.save(hexperName..'/cond_lossFun.t7', lossFunTrack)

   -- -- plot the loss function track
   -- gnuplot.epsfigure(hexperName.. '/lossFactor.eps')
   -- gnuplot.title('Loss function minimization over epochs')
   -- gnuplot.plot(lossFunTrack)   
   -- gnuplot.xlabel('epochs')
   -- gnuplot.ylabel('L(x)')
   -- -- close the gnuplot files
   -- gnuplot.plotflush()
   -- -- step out the current directory

end
