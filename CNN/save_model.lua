----------------------------------------------------------------------
-- save models so that it can be read on Windows
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

bank1 = torch.load('results/bank1.net')
bank2 = torch.load('results/bank2.net')
bank3 = torch.load('results/bank3.net')
linear_nn = torch.load('results/linear_nn.net')

local jtorch = dofile("../jtorch/jtorch.lua")
jtorch.init("../jtorch/")
jtorch.saveModel(bank1, "xy_bank1.bin")
jtorch.saveModel(bank2, "xy_bank2.bin")
jtorch.saveModel(bank3, "xy_bank3.bin")
jtorch.saveModel(linear_nn, "xy_linear_nn.bin")
