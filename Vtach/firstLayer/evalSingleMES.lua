require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'
-- fix the seed

opt = { 
   testData = true,
   evalVT = false,
   MaxNumEpoches = 300,
   optimization = 'LBFGS', -- 'LBFGS',
   learningRate = 0.01,
   momentum = 0.95,
   weightDecay = 0.01,
   RMSalpha = 0.9,   -- RMSprop only
   maxIter = 5,
   numThreads = 1
}

os.execute('killall gnuplot')

j,k,nt,nc=1,1,1,1
hyperExperName = '/home/mark/Documents/VT_learning/Logs/logPAsquare/MES_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
factor = torch.load(hyperExperName.."/factor.net")

-- load the data and trained model
qrsData = torch.load('/home/mark/Documents/VT_learning/data/uniformSample/qrsData1012_test_sr3.t7')
-- transform the input data into cuda
qrsData.train_x = qrsData.train_x:cuda()
qrsData.train_y = qrsData.train_y:cuda()
qrsData.train_coord = qrsData.train_coord:cuda()
qrsData.val_x = qrsData.val_x:cuda()
qrsData.val_y = qrsData.val_y:cuda()
qrsData.val_coord = qrsData.val_coord:cuda()
flowNet = factor:get(1):get(1):get(1)
flowNet_val = flowNet:clone('weight','bias')
--print('The structure of flow Net:\n'..flowNet:__tostring())

--propagate data, training data and validation dataset
train_nodes = flowNet:forward(qrsData.train_x)
val_nodes = flowNet_val:forward(qrsData.val_x)


flatteningVTPA = nn.JoinTable(2)
flatteningVTPA_val = nn.JoinTable(2)
flatteningVTPA:cuda()
flatteningVTPA_val:cuda()
train_flatten_nodes = flatteningVTPA:forward(train_nodes)
val_flatten_nodes = flatteningVTPA_val:forward(val_nodes)

-- evaluate the learned features with with a classifier
numHiddenUnits_VT = train_nodes[1]:size(2)
numHiddenUnits = train_flatten_nodes:size(2)
-- classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'}
-- numLabels = #classes
numCoordDim = 3

eval = nn.Sequential()
if opt.evalVT then
   eval:add(nn.Linear(numHiddenUnits_VT,numCoordDim))
   inputs = train_nodes[1]
   vals = val_nodes[1]
else
   eval:add(nn.Linear(numHiddenUnits,numCoordDim))
   inputs = train_flatten_nodes
   vals = val_flatten_nodes
end


-- eval:add(nn.LogSoftMax())
eval:cuda()

parameters,gradParameters = eval:getParameters()

dataNum_train = inputs:size(1)
dataNum_val = vals:size(1)


-- criterion = nn.ClassNLLCriterion()
criterion = nn.MSECriterion()

criterion:cuda()


--confusion = optim.ConfusionMatrix(classes)
--confusion_val = optim.ConfusionMatrix(classes)
eval_val = eval:clone('weight','bias')

epoch = 0


while epoch < opt.MaxNumEpoches do
   
   xlua.progress(epoch, opt.MaxNumEpoches)

   epoch = epoch + 1
   local time = sys.clock()
   local mser_train = 0
   
   local MSERloss = function(outputs, y)
      return torch.sqrt(torch.sum(torch.pow(outputs-y,2),2)):mean()
   end

   local feval =function (x)
      collectgarbage()

      -- get new parameters
      if x ~= parameters then
	 parameters:copy(x)
      end
      
      -- reset the gradients
      gradParameters:zero()

      -- forward the training data
      local outputs = eval:forward(inputs)
      local f = criterion:forward(outputs,qrsData.train_coord)
      mser_train = MSERloss(outputs,qrsData.train_coord)
      local df = criterion:backward(outputs,qrsData.train_coord)
      eval:backward(inputs, df)
      
      -- update confusion matrix for training data
--      for i =1 , dataNum_train do
--	 confusion:add(outputs[i],qrsData.train_y[i][1])
--      end
      
      -- weight penlties
      if opt.weightDecay ~=0 and opt.optimization ~= 'SGD' then
	 local norm = torch.norm
	 f = f + opt.weightDecay* norm(parameters,2)^2/2

	 -- update gradients
	 gradParameters:add( parameters:clone():mul(opt.weightDecay))
      end

      return f, gradParameters
      
   end

   --evaluate the trainined model on validation set
   local outputs_val = eval_val:forward(vals)
   local f_val = criterion:forward(outputs_val,qrsData.val_coord)
   local mser_val = MSERloss(outputs_val,qrsData.val_coord)


   -- update confusion matrix for validation dataset
--   for i =1 , dataNum_val do
--      confusion_val:add(outputs_val[i],qrsData.val_y[i][1])
--   end
   
  


   -- optimize on current mini-batch
   if opt.optimization == 'LBFGS' then

      -- Perform LBFGS step:
      local lbfgsState = lbfgsState or {
	 maxIter = opt.maxIter,
	 learningRate = opt.learningRate
				       }
      x,lossFun= optim.lbfgs(feval, parameters, lbfgsState)
      

   elseif opt.optimization == 'SGD' then
      -- Perform SGD step:
      local sgdState = sgdState or {
	 learningRate = opt.learningRate,
	 momentum = opt.momentum,
	 weightDecay = opt.weightDecay,
	 learningRateDecay = 5e-7
				   }
      x,lossFun = optim.sgd(feval, parameters, sgdState)
   elseif opt.optimization == 'RMSprop' then
      local rmspropConfig = rmspropConfig or {
	 learningRate = opt.learningRate,
	 alpha = opt.RMSalpha
					     }
      x,lossFun = optim.rmsprop(feval, parameters, rmspropConfig)
   else
      error('unknown optimization method')
   end

   if epoch > 1 then
      lossFunTrack = torch.cat(lossFunTrack, torch.Tensor(1,1):fill(torch.sqrt(lossFun[1])),1)
      lossFunTrack_val = torch.cat(lossFunTrack_val, torch.Tensor(1,1):fill(torch.sqrt(f_val)),1)
      mserFunTrack_train = torch.cat(mserFunTrack_train, torch.Tensor(1,1):fill(mser_train),1)
      mserFunTrack_val = torch.cat(mserFunTrack_val, torch.Tensor(1,1):fill(mser_val),1)
   else
      lossFunTrack = torch.Tensor(1,1):fill(torch.sqrt(lossFun[1]))
      lossFunTrack_val = torch.Tensor(1,1):fill(torch.sqrt(f_val))
      mserFunTrack_train = torch.Tensor(1,1):fill(mser_train)
      mserFunTrack_val =  torch.Tensor(1,1):fill(mser_val)
   end

   
   -- tracking down the training and validation set classification accuracies
--   confusion:updateValids()
--   train_acc = confusion.totalValid * 100
--   if epoch > 1 then
--      train_acc_track = torch.cat(train_acc_track, torch.Tensor(1,1):fill(train_acc),1)
--   else
--      train_acc_track = torch.Tensor(1,1):fill(train_acc)
--   end
--   confusion:zero()

   -- calculate the confusion  accuracy for validation set
--   confusion_val:updateValids()
--   val_acc = confusion_val.totalValid * 100
--   if epoch > 1 then
--     val_acc_track = torch.cat(val_acc_track, torch.Tensor(1,1):fill(val_acc),1)
--   else
--      val_acc_track = torch.Tensor(1,1):fill(val_acc)
--   end

--   confusion_val:zero()

   -- time taken
   time = sys.clock() - time
   --   print("<trainer> time to learn whole dataset = " .. (time) .. 'ms')

end

numSamplesForMargin = 4000
margin_index= torch.randperm(train_nodes[1]:size(1))[{{1,2*numSamplesForMargin}}]
margin_vt_pair1 = torch.Tensor(numSamplesForMargin, train_nodes[1]:size(2)):cuda()
margin_vt_pair2 = torch.Tensor(numSamplesForMargin, train_nodes[1]:size(2)):cuda()
margin_pa_pair1 = torch.Tensor(numSamplesForMargin, train_nodes[2]:size(2)):cuda()
margin_pa_pair2 = torch.Tensor(numSamplesForMargin, train_nodes[2]:size(2)):cuda()

for i=1,numSamplesForMargin do
   margin_vt_pair1[i] = train_nodes[1][margin_index[i]]
   margin_vt_pair2[i] = train_nodes[1][margin_index[i+numSamplesForMargin]]
   margin_pa_pair1[i] = train_nodes[2][margin_index[i]]
   margin_pa_pair2[i] = train_nodes[2][margin_index[i+numSamplesForMargin]]
end
vt_distance = torch.sqrt(torch.norm(margin_vt_pair1 - margin_vt_pair2,2)^2/numSamplesForMargin)
pa_distance = torch.sqrt(torch.norm(margin_pa_pair1 - margin_pa_pair2,2)^2/numSamplesForMargin)

print("Average VT distance is "..tostring(vt_distance))
print("Average PA distance is "..tostring(pa_distance))
-- plot the loss function track
lossFactor = torch.load(hyperExperName.."/factor_lossFun.t7")
optFactor = torch.load(hyperExperName.."/hypers.t7")
print(optFactor)
-- save the loss function and classification track
losses = {
   lossTrain = lossFunTrack,
   lossVal = lossFunTrack_val,
   mserTrain = mserFunTrack_train,
   mserVal = mserFunTrack_val,
}
if opt.evalVT then
   os.execute('mkdir -p '.. hyperExperName..'/VTnodes')
   torch.save(hyperExperName..'/VTnodes/classifyTrack.t7',losses)
else
   torch.save(hyperExperName..'/classifyTrack.t7',losses)
end
--plot them
--gnuplot.epsfigure(hyperExperName.."/factorLoss.eps")
  gnuplot.figure(1)
gnuplot.title('Factor Loss function minimization over epochs')
gnuplot.plot(lossFactor)   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')
print('the loss of factor is '..tostring(torch.min(lossFactor)))
print('the minimum loss distance is '..tostring(torch.min(mserFunTrack_val)))
--gnuplot.epsfigure(hyperExperName.."/classfy.eps")
 gnuplot.figure(2)
gnuplot.title('Coordinates regression loss function minimization over epochs')
gnuplot.plot({'Training',lossFunTrack,'-'}, {'Validation',lossFunTrack_val,'-'})   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')

 gnuplot.figure(3)
gnuplot.title('Geodestic distance minimization over epochs')
gnuplot.plot({'Training',mserFunTrack_train,'-'}, {'Validation',mserFunTrack_val,'-'})   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')
