-- Actually you can total write a shell script to do such stuff.
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'cunn'

tranCudaData = {} 

function tranCudaData.run(j, k, nt, nc)
   
   hyperExperName_load = '/home/mark/Documents/VT_learning/Logs/squarePAlayer2/MES_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
   hyperExperName_save = '/home/mark/Documents/VT_learning/Logs/squarePAlayer2_trans/MES_'..'j'..tostring(j)..'k'..tostring(k)..'nt'..tostring(nt)..'nc'..tostring(nc)
   lossFactor = torch.load(hyperExperName_load..'/cond_lossFun_vt.t7')
   classifyTrack = torch.load(hyperExperName_load..'/classifyTrack.t7')
  

   os.execute('mkdir -p '.. hyperExperName_save)
   torch.save(hyperExperName_save..'/cond_lossFun_vt.t7', lossFactor)
   torch.save(hyperExperName_save..'/classifyTrack.t7', classifyTrack)

end
   
