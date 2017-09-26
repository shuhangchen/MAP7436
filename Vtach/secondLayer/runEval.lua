
require 'evalCond'
print('Eval for trained conditional auto encoders  has begin.\n')
for j=1,2 do
   for k=1,2 do
      for nt=1,2 do
	 for nc = 1,3 do
	    
--	    evalCond.eval(j,k,nt,nc,true)
	    evalCond.eval(j,k,nt,nc,false)
	    
	    collectgarbage()
	 end
      end
   end
end

print('Eval for trained Factor parameters has finished.\n')
      
