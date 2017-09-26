-- flow the net with input from first layer (MES) to generate input for current data

require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'
-- fix the seed


hyperExperName = '/home/mark/Documents/VT_learning/coordinates/thirdLayer/candidates'

-- load the data and trained model
cond= torch.load(hyperExperName.."/cond.net")
qrsData = torch.load('/home/mark/Documents/VT_learning/coordinates/thirdLayer/qrsLayer3.t7')


flowNet = cond:get(1):get(1):get(1)
print(tostring(flowNet))
flowNet_val = flowNet:clone('weight','bias')
flowNet_test = flowNet:clone('weight','bias')
--print('The structure of flow Net:\n'..flowNet:__tostring())


--propagate data, training data and validation dataset
qrsLayer4 = {
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
--torch.save(hyperExperName..'/cond_full.net',flowNet)
torch.save('/home/mark/Documents/VT_learning/coordinates/thirdLayer/qrsLayer4.t7',qrsLayer4)
