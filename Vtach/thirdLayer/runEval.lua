
require 'evalCond'
print('Eval for trained conditional auto encoders  has begin.\n')
for i=1,1 do
   for j=1,3 do
      for k=1,4 do
	 for nt=1,1 do
	    for nc=1,5 do
	       
	       evalCond.eval(i,j,k,nt,nc)
	       
	       collectgarbage()
	    end
	 end
      end
   end
end
print('Eval for trained Factor parameters has finished.\n')
      
