
-- this file opens the saved opts and lossFun/accuracy track,
-- show the parameters, and plot the track data

require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'


os.execute('killall gnuplot')
--i,j,k,nt,nc = 2,2,1,1,2
--hyperExperName = '/home/mark/Documents/VT_learning/Logs/logMES/hexLog_i'.. tostring(i)..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
-- 3,1,1,4
j,k,nt,nc=3,1,1,4

hyperExperName = '/home/mark/Documents/VT_learning/Logs/squareThirdLayer/MES_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)

--hyperExperName = '/media/usb/log/MES/MES_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
optFactor = torch.load(hyperExperName.."/hypers.t7")

print("optFactor is: ")
print(optFactor)


lossFactor_vt = torch.load(hyperExperName.."/cond_lossFun.t7")



lossEval = torch.load(hyperExperName.."/classifyTrack.t7")
print('The best loss for full nodes is '..tostring(torch.min(lossEval.lossVal)))
print('The best distance for full nodes is '..tostring(torch.min(lossEval.mserVal)))
--plot them
gnuplot.figure(1)
gnuplot.title('Factor Loss function minimization over epochs')
gnuplot.plot(lossFactor_vt)   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')


-- plot the accuracy function track
gnuplot.figure(2)
gnuplot.title('Classification loss function minimization over epochs for full nodes')
gnuplot.plot({'Training',lossEval.lossTrain,'-'}, {'Validation',lossEval.lossVal,'-'})   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')



gnuplot.figure(3)
gnuplot.title('Geodestic distance minimization over epochs for full nodes')
gnuplot.plot({'Training',lossEval.mserTrain,'-'}, {'Validation',lossEval.mserVal,'-'})   
gnuplot.xlabel('epochs')
gnuplot.ylabel('L(x)')
