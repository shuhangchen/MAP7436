
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'
-- fix the seed

opt = {
   MaxNumEpoches = 500,
   optimization = 'LBFGS', -- 'LBFGS',
   learningRate = 0.01,
   momentum = 0.95,
   weightDecay = 0,
   RMSalpha = 0.9,   -- RMSprop only
   maxIter = 5,
   trainVT  = true,
   numThreads = 1
}


os.execute('killall gnuplot')

--   j,k,nc=1,1,1
hyperExperName = '/home/mark/Documents/VT_learning/squarePA/thirdLayer/candidates'

--hyperExperName = '/home/mark/Documents/VT_learning/Logs/logBEC/BEC_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
factor = torch.load(hyperExperName.."/cond.net")


-- load the data and trained model
-- qrsData = torch.load('/home/mark/Documents/VT_learning/data/qrsDataNormalized.t7')
qrsData_loaded = torch.load('/home/mark/Documents/VT_learning/squarePA/secondLayer/candidates/qrsLayer2.t7')
-- transform the input data into cuda
qrsData = {
   train_x = {qrsData_loaded.train_x[1]:clone(),qrsData_loaded.train_x[2]:clone()},
   train_y = qrsData_loaded.train_y:clone(),
   train_coord = qrsData_loaded.train_coord:clone(),
   val_x = {qrsData_loaded.val_x[1]:clone(), qrsData_loaded.val_x[2]:clone()},
   val_y = qrsData_loaded.val_y:clone(),
   val_coord = qrsData_loaded.val_coord:clone(),
   test_x = {qrsData_loaded.test_x[1]:clone(), qrsData_loaded.test_x[2]:clone()},
   test_y = qrsData_loaded.test_y:clone(),
   test_coord = qrsData_loaded.test_coord:clone()
}
if not opt.trainVT then
   -- we are training the conditional factor model for VT nodes
   qrsData.train_x[1],qrsData.train_x[2] = qrsData_loaded.train_x[2], qrsData_loaded.train_x[1]
   qrsData.val_x[1],qrsData.val_x[2] = qrsData_loaded.val_x[2], qrsData_loaded.val_x[1]
   qrsData.test_x[1],qrsData.test_x[2] = qrsData_loaded.test_x[2], qrsData_loaded.test_x[1]
end
flowNet = factor:get(1):get(1):get(1)
flowNet_val = flowNet:clone('weight','bias')
flowNet_test = flowNet:clone('weight','bias')
--print('The structure of flow Net:\n'..flowNet:__tostring())

--propagate data, training data and validation dataset
train_nodes = flowNet:forward(qrsData.train_x)
val_nodes = flowNet_val:forward(qrsData.val_x)
test_node = flowNet_test:forward(qrsData.test_x)

-- evaluate the learned features with with a classifier
numHiddenUnits = train_nodes:size(2)
-- classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'}
numCoordDim = 3


eval = nn.Sequential()

eval:add(nn.Linear(numHiddenUnits,numCoordDim))
inputs = train_nodes
vals = val_nodes

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

local MSERloss = function(outputs, y)
   MSERstat = torch.sqrt(torch.sum(torch.pow(outputs-y,2),2))
   MSERxyz={
      x= torch.abs(outputs[{{},{1}}]-y[{{},{1}}]):mean(),
      y= torch.abs(outputs[{{},{2}}]-y[{{},{2}}]):mean(),
      z= torch.abs(outputs[{{},{3}}]-y[{{},{3}}]):mean()
   }
   MSERmean = MSERstat:mean()
   MSERstd = MSERstat:std()
   CI95 = 1.96*MSERstd/torch.sqrt(outputs:size(1))
   return MSERmean,MSERstd,CI95,MSERxyz
end

epoch = 0


while epoch < opt.MaxNumEpoches do
   
   xlua.progress(epoch, opt.MaxNumEpoches)

   epoch = epoch + 1
   local time = sys.clock()
   local mser_train = 0
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
      mserFunTrack_train = torch.cat(mserFunTrack_train, torch.Tensor(1,1):fill(mser_train),1)
      mserFunTrack_val = torch.cat(mserFunTrack_val, torch.Tensor(1,1):fill(mser_val),1)
   else
      lossFunTrack = torch.Tensor(1,1):fill(torch.sqrt(lossFun[1]))
      lossFunTrack_val = torch.Tensor(1,1):fill(torch.sqrt(f_val))
      mserFunTrack_train = torch.Tensor(1,1):fill(mser_train)
      mserFunTrack_val =  torch.Tensor(1,1):fill(mser_val)
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

bestRecord,bestIndice = torch.min(mserFunTrack_val,1)
print('The smalles validation distance error is '..tostring(bestRecord)..' at '..tostring(bestIndice))




local outputs_train = eval:forward(train_nodes)
local f_train = criterion:forward(outputs_train,qrsData.train_coord)
local mser_train_mean,mser_train_std,mser_train_ci95 = MSERloss(outputs_train,qrsData.train_coord)

local outputs_val = eval_val:forward(val_nodes)
local f_val = criterion:forward(outputs_val,qrsData.val_coord)
local mser_val_mean,mser_val_std,mser_val_ci95 = MSERloss(outputs_val,qrsData.val_coord)

local outputs_test = eval_test:forward(test_node)
local f_test = criterion:forward(outputs_test,qrsData.test_coord)
local mser_test_mean,mser_test_std,mser_test_ci95 = MSERloss(outputs_test,qrsData.test_coord)

print('The stat on train dataset is mean error: '..tostring(mser_train_mean)..', the std: '..tostring(mser_train_std)..', the 95% confidence interval: '..tostring(mser_train_ci95))
print('The stat on val dataset is mean error: '..tostring(mser_val_mean)..', the std: '..tostring(mser_val_std)..', the 95% confidence interval: '..tostring(mser_val_ci95))
print('The stat on test dataset is mean error: '..tostring(mser_test_mean)..', the std: '..tostring(mser_test_std)..', the 95% confidence interval: '..tostring(mser_test_ci95))


--plot them
-- we can not plot them in terminal of the remote server
--   gnuplot.epsfigure(hyperExperName.."/factorLoss.eps")
--   gnuplot.title('Factor Loss function minimization over epochs')
--   gnuplot.plot(lossFactor)   
--   gnuplot.xlabel('epochs')
--   gnuplot.ylabel('L(x)')
-- --  print('the loss of factor is '..tostring(torch.min(lossFactor)))
--   gnuplot.epsfigure(hyperExperName.."/classfy.eps")
--   gnuplot.title('Classification loss function minimization over epochs')
--   gnuplot.plot({'Training',lossFunTrack,'-'}, {'Validation',lossFunTrack_val,'-'})   
--   gnuplot.xlabel('epochs')
--   gnuplot.ylabel('L(x)')
--   gnuplot.plotflush()

--   gnuplot.epsfigure(hyperExperName.."/distances.eps")
--   gnuplot.title('Geodestic distance minimization over epochs')
--   gnuplot.plot({'Training',mserFunTrack_train,'-'}, {'Validation',mserFunTrack_val,'-'})   
--   gnuplot.xlabel('epochs')
--   gnuplot.ylabel('L(x)')

