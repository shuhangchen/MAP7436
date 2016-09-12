# Fast bundle level methods for unconstrained and ball-constrained convex optimization
This is a [torch](http://torch.ch/)-based implementation of the fastAPL algorithm whic could be found [here](http://arxiv.org/abs/1412.2128). Author's webpage also contains the original [Matlab codes](http://arxiv.org/abs/1412.2128).

## miscellaneous
* I have only implemented the fastAPL method, and I will try to finish the other bundle type method, fast USL, recently.
* Current version only has tests for solving qurdratic programming problems.
Here is the function value tracking during iteration schemes
![alt text](https://rawgit.com/shuhangchen/MAP7436/master/bundleLevel/fValueTrack.png)
* This version of implementation is highly unstable, completely unoptimized. I will try to fix these problems very recently.


