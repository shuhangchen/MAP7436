
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'



opt = {
   evalVT = false,
   MaxNumEpoches = 300,
   optimization = 'LBFGS', -- 'LBFGS',
   learningRate = 0.02,
   momentum = 0.95,
   weightDecay = 0,
   RMSalpha = 0.9,   -- RMSprop only
   maxIter = 5,
   trainVT  = true,
   numThreads = 1
}


os.execute('killall gnuplot')

j,k,nc=1,1,1
hyperExperName = '/home/mark/Documents/VT_learning/Logs/logMES_coord_cond_full/MES_'..'j'..tostring(j)..'k'..tostring(k)..'nc'..tostring(nc)
--hyperExperName = '/home/mark/Documents/VT_learning/Logs/logBEC/BEC_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)

hypers = torch.load(hyperExperName.."/hypers.t7")

-- load the data and trained model
-- qrsData = torch.load('/home/mark/Documents/VT_learning/data/qrsDataNormalized.t7')
qrsData = torch.load('/home/mark/Documents/VT_learning/coordinates/secondLayer/qrsLayer2_test.t7')
-- transform the input data into cuda

cond_VT = torch.load(hyperExperName..'/cond_VT.net')

cond_PA = torch.load(hyperExperName..'/cond_PA.net')


flowNet = nn.ConcatTable():add(cond_VT:get(1):get(1):get(1)):add(cond_PA:get(1):get(1):get(1) )
flowNet_val = flowNet:clone('weight','bias')
flowNet_test = flowNet:clone('weight','bias')
--print('The structure of flow Net:\n'..flowNet:__tostring())



--propagate data, training data and validation dataset
train_nodes = flowNet:forward(qrsData.train_x)
val_nodes = flowNet_val:forward(qrsData.val_x)
test_nodes = flowNet_test:forward(qrsData.test_x)

flatteningVTPA = nn.JoinTable(2)
flatteningVTPA_val = nn.JoinTable(2)
flatteningVTPA_test = nn.JoinTable(2)
flatteningVTPA:cuda()
flatteningVTPA_val:cuda()
flatteningVTPA_test:cuda()
train_flatten_nodes = flatteningVTPA:forward(train_nodes)
val_flatten_nodes = flatteningVTPA_val:forward(val_nodes)
test_flatten_nodes  =flatteningVTPA_test:forward(test_nodes)

-- evaluate the learned features with with a classifier
numHiddenUnits_VT = train_nodes[1]:size(2)
numHiddenUnits = train_flatten_nodes:size(2)
-- classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'}
numCoordDim = 3


eval = nn.Sequential()

if opt.evalVT then
   eval:add(nn.Linear(numHiddenUnits_VT,numCoordDim))
   inputs = train_nodes[1]
   vals = val_nodes[1]
   tests = test_nodes[1]
else
   eval:add(nn.Linear(numHiddenUnits,numCoordDim))
   inputs = train_flatten_nodes
   vals = val_flatten_nodes
   tests = test_flatten_nodes
end

-- eval:add(nn.LogSoftMax())
eval:cuda()

parameters,gradParameters = eval:getParameters()

dataNum_train = inputs:size(1)
dataNum_val = vals:size(1)


--  criterion = nn.ClassNLLCriterion()
criterion = nn.MSECriterion()
criterion:cuda()


--   confusion = optim.ConfusionMatrix(classes)
--   confusion_val = optim.ConfusionMatrix(classes)
eval_val = eval:clone('weight','bias')
eval_test = eval:clone('weight','bias')
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
      -- for i =1 , dataNum_train do
      --    confusion:add(outputs[i],qrsData.train_y[i][1])
      -- end
      
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

   local outputs_test = eval_test:forward(tests)
   local f_test = criterion:forward(outputs_test,qrsData.test_coord)
   local mser_test = MSERloss(outputs_test,qrsData.test_coord)
   -- update confusion matrix for validation dataset
   -- for i =1 , dataNum_val do
   -- 	 confusion_val:add(outputs_val[i],qrsData.val_y[i][1])
   -- end

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
      lossFunTrack_test = torch.cat(lossFunTrack_test, torch.Tensor(1,1):fill(torch.sqrt(f_test)),1)
      mserFunTrack_train = torch.cat(mserFunTrack_train, torch.Tensor(1,1):fill(mser_train),1)
      mserFunTrack_val = torch.cat(mserFunTrack_val, torch.Tensor(1,1):fill(mser_val),1)
      mserFunTrack_test = torch.cat(mserFunTrack_test, torch.Tensor(1,1):fill(mser_test),1)
   else
      lossFunTrack = torch.Tensor(1,1):fill(torch.sqrt(lossFun[1]))
      lossFunTrack_val = torch.Tensor(1,1):fill(torch.sqrt(f_val))
      lossFunTrack_test = torch.Tensor(1,1):fill(torch.sqrt(f_test))
      mserFunTrack_train = torch.Tensor(1,1):fill(mser_train)
      mserFunTrack_val =  torch.Tensor(1,1):fill(mser_val)
      mserFunTrack_test =  torch.Tensor(1,1):fill(mser_test)
   end

   
   -- tracking down the training and validation set classification accuracies
   -- confusion:updateValids()
   -- train_acc = confusion.totalValid * 100
   -- if epoch > 1 then
   -- 	 train_acc_track = torch.cat(train_acc_track, torch.Tensor(1,1):fill(train_acc),1)
   -- else
   -- 	 train_acc_track = torch.Tensor(1,1):fill(train_acc)
   -- end
   -- confusion:zero()

   -- -- calculate the confusion  accuracy for validation set
   -- confusion_val:updateValids()
   -- val_acc = confusion_val.totalValid * 100
   -- if epoch > 1 then
   -- 	 val_acc_track = torch.cat(val_acc_track, torch.Tensor(1,1):fill(val_acc),1)
   -- else
   -- 	 val_acc_track = torch.Tensor(1,1):fill(val_acc)
   -- end

   -- confusion_val:zero()

   -- time taken
   time = sys.clock() - time
   --   print("<trainer> time to learn whole dataset = " .. (time) .. 'ms')

end


-- plot the loss function track
loss_VT = torch.load(hyperExperName..'/cond_lossFun_vt.t7')
loss_PA = torch.load(hyperExperName..'/cond_lossFun_PA.t7')

-- save the loss function and classification track
losses = {
   lossTrain = lossFunTrack,
   lossVal = lossFunTrack_val
}
torch.save(hyperExperName..'/classifyTrack.t7',losses)

--plot them
gnuplot.epsfigure(hyperExperName.."/factorLoss.eps")
gnuplot.title('Factor Loss function minimization over epochs')
gnuplot.plot(lossFactor)   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')
--  print('the loss of factor is '..tostring(torch.min(lossFactor)))
gnuplot.epsfigure(hyperExperName.."/classfy.eps")
gnuplot.title('Classification loss function minimization over epochs')
gnuplot.plot({'Training',lossFunTrack,'-'}, {'Validation',lossFunTrack_val,'-'})   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')
gnuplot.plotflush()

gnuplot.epsfigure(hyperExperName.."/distances.eps")
gnuplot.title('Geodestic distance minimization over epochs')
gnuplot.plot({'Training',mserFunTrack_train,'-'}, {'Validation',mserFunTrack_val,'-'})   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')

print('the minimum loss distance is '..tostring(torch.min(mserFunTrack_test)))
