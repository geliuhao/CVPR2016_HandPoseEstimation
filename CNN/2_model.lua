----------------------------------------------------------------------
-- 2. define CNN model
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- input dimensions
nfeats = 1
width = 96
height = 96
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {16,32,6804}
linear_input_num = 7776
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

-- opt.model == 'convnet'

-- Bank 1
bank1 = nn.Sequential()
	  -- stage 1 : Convolution -> ReLU -> Max pooling
	  filtsize = 5
	  poolsize = 4
	  bank1:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      bank1:add(nn.ReLU())
      bank1:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
      -- stage 2 : Convolution -> ReLU -> Max pooling
	  filtsize = 6 --2	--6
	  poolsize = 2
      bank1:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      bank1:add(nn.ReLU())
      bank1:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
      --bank1:add(nn.SpatialAdaptiveMaxPooling(9,9))

-- Bank 2
bank2 = nn.Sequential()
	  -- stage 1 : Convolution -> ReLU -> Max pooling
	  filtsize = 5
	  poolsize = 2
	  bank2:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      bank2:add(nn.ReLU())
      bank2:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
      -- stage 2 : Convolution -> ReLU -> Max pooling
	  filtsize = 5
	  poolsize = 2
      bank2:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      bank2:add(nn.ReLU())
      bank2:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- Bank 3
bank3 = nn.Sequential()
	  -- stage 1 : Convolution -> ReLU -> Max pooling
	  filtsize = 4
	  poolsize = 1
	  bank3:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      bank3:add(nn.ReLU())
      bank3:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
      -- stage 2 : Convolution -> ReLU -> Max pooling
	  filtsize = 4
	  poolsize = 2
      bank3:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      bank3:add(nn.ReLU())
      bank3:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- 2-stage neural network
linear_nn = nn.Sequential()
	  linear_nn:add(nn.Reshape(linear_input_num))
      linear_nn:add(nn.Linear(linear_input_num, nstates[3]))
      linear_nn:add(nn.ReLU())
      linear_nn:add(nn.Linear(nstates[3], noutputs))

----------------------------------------------------------------------
if opt.visualize then
	print '==> here is the 1st bank:'
	print(bank1)
	print '==> here is the 2nd bank:'
	print(bank2)
	print '==> here is the 3rd bank:'
	print(bank3)
	print '==> here is the 2-stage neural network:'
	print(linear_nn)
end

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
	if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Bank 1 - Layer 1 filters:')
	 itorch.image(bank1:get(1).weight)
	 print('Bank 1 - Layer 2 filters:')
	 itorch.image(bank1:get(4).weight)
	 print('Bank 2 - Layer 1 filters:')
	 itorch.image(bank2:get(1).weight)
	 print('Bank 2 - Layer 2 filters:')
	 itorch.image(bank2:get(4).weight)
	 print('Bank 3 - Layer 1 filters:')
	 itorch.image(bank3:get(1).weight)
	 print('Bank 3 - Layer 2 filters:')
	 itorch.image(bank3:get(4).weight)
	 
	 --print('Linear NN - Layer 1 filters:')
	 --itorch.image(linear_nn:get(2).weight)
	 --print('Linear NN - Layer 2 filters:')
	 --itorch.image(linear_nn:get(4).weight)
   end
end
