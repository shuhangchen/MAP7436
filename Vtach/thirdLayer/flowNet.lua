-- flow the net with input from first layer (MES) to generate input for current data

require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'
-- fix the seed


hyperExperName = '/home/mark/Documents/VT_learning/squarePA/secondLayer/candidates'

-- load the data and trained model
cond_VT= torch.load(hyperExperName.."/cond_VT.net")
cond_PA= torch.load(hyperExperName.."/cond_PA.net")
qrsData = torch.load('/home/mark/Documents/VT_learning/squarePA/firstLayer/candidates/qrsLayer1.t7')


flowNet = nn.ConcatTable():add(cond_VT:get(1):get(1):get(1)):add(cond_PA:get(1):get(1):get(1))
print(tostring(flowNet))
flowNet_val = flowNet:clone('weight','bias')
flowNet_test = flowNet:clone('weight','bias')
--print('The structure of flow Net:\n'..flowNet:__tostring())


--propagate data, training data and validation dataset
qrsLayer3 = {
   train_x = flowNet:forward(qrsData.train_x),
   val_x = flowNet_val:forward(qrsData.val_x),
   test_x = flowNet_test:forward(qrsData.test_x),
   train_y = qrsData.train_y,
   val_y = qrsData.val_y,
   test_y = qrsData.test_y,
   train_coord = qrsData.train_coord,
   val_coord = qrsData.val_coord,
   test_coord = qrsData.test_coord
}
torch.save(hyperExperName..'/cond_full.net',flowNet)
torch.save('/home/mark/Documents/VT_learning/squarePA/secondLayer/candidates/qrsLayer2.t7',qrsLayer3)
