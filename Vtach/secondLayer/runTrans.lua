
require 'tranCudaData'

for j=1,3 do
   for k=1,3 do
      for nt=1,2 do
	 for nc = 1,3 do
	    tranCudaData.run(j,k,nt,nc)
	    collectgarbage()
	 end
      end
   end
end


