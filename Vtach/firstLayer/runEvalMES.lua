require 'evalFactorMES'
print('Eval for trained factor MES parameters  has begin.\n')
for j=1,2 do
   for k=1,3 do
      for nt=1,1 do
	 for nc=1,4 do
	    
	    evalFactorMES.eval(j,k,nt,nc)
	    
	    collectgarbage()
	 end
      end
   end
end

print('Eval for trained factor MSE parameters has finished.\n')
