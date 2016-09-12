require 'fastAPL'
local matio = require 'matio'
local size = 10
local bundleSize = 5
local x = torch.randn(size)
local bundles = {
   coef_A = torch.randn(bundleSize, size),
   coef_b = torch.randn(bundleSize),
   size = 5
}
local proj = {
   coef_a = torch.randn(size),
   coef_b = torch.randn(1)
}
local new_cuttingPlane = {
   coef_a = torch.randn(size),
   coef_b = torch.randn(1)
}
local proxCenter = torch.Tensor(size):zero()
local lowerBound = 1.0
local bundleLevel = 2.0

local found, solution = fastAPL.proxMapping_kkt(x,bundles, proj, new_cuttingPlane, proxCenter, lowerBound, bundleLevel)
print(found)
print(solution)
matio.save('proxTest.mat',{bundle_coef_a = bundles.coef_A, bundle_coef_b = bundles.coef_b, proj_coef_a = proj.coef_a, proj_coef_b = proj.coef_b, new_coef_a = new_cuttingPlane.coef_a, new_coef_b = new_cuttingPlane.coef_b, proxCenter = proxCenter, solution = solution})

-- the proxmapping has the same results as the ProxMapping_kkt function, so assume now this facility is right
