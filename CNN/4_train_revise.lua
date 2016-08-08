----------------------------------------------------------------------
-- 4. training
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 0.2, 'learning rate at t=0')--1e-3
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0.0005, 'weight decay (SGD only)')--0
   cmd:option('-momentum', 0.9, 'momentum (SGD only)')--0
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:option('-experiment_index', 2, 'experiment index')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   bank1:cuda()
   bank2:cuda()
   bank3:cuda()
   linear_nn:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
--classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
--confusion = optim.ConfusionMatrix(classes)

-- Mean-square error
mean_square_error = 0.0

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, opt.experiment_index, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, opt.experiment_index, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector

if bank1 and bank2 and bank3 and linear_nn then
   parameters_bank1,gradParameters_bank1 = bank1:getParameters()
   parameters_bank2,gradParameters_bank2 = bank2:getParameters()
   parameters_bank3,gradParameters_bank3 = bank3:getParameters()
   parameters_linear,gradParameters_linear = linear_nn:getParameters()
   num_bank1 = (#parameters_bank1)[1]
   num_bank2 = (#parameters_bank2)[1]
   num_bank3 = (#parameters_bank3)[1]
   num_liear = (#parameters_linear)[1]
   num_parameters = num_bank1 + num_bank2 + num_bank3 + num_liear
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 0	--1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   --model:training()
   bank1:training()
   bank2:training()
   bank3:training()
   linear_nn:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   train_counter = 0
   for t = 1,trsize,opt.batchSize do
      -- disp progress
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs1 = {}
      local inputs2 = {}
      local inputs3 = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trsize) do
         -- rotate and load new sample
		 if trValid[{shuffle[i],1}]~=0 then
			 train_counter = train_counter + 1

			 local rotate_rad = (torch.rand(1)*16-8)/180*3.1415926
		     local input1 = image.rotate(train_im_96[{{shuffle[i]}}], rotate_rad[1])
			 local input2 = image.rotate(train_im_48[{{shuffle[i]}}], rotate_rad[1])
			 local input3 = image.rotate(train_im_24[{{shuffle[i]}}], rotate_rad[1])
			--]]
		     local temp = image.rotate(trlabels[shuffle[i]], rotate_rad[1])
	--[[
		     local input1 = train_im_96[{{shuffle[i]}}]
			 local input2 = train_im_48[{{shuffle[i]}}]
			 local input3 = train_im_24[{{shuffle[i]}}]
			 local temp = trlabels[shuffle[i]]
	--]]
			 local target = nn.Reshape(noutputs):forward(temp)
		     if opt.type == 'double' then
				input1 = input1:double()
				input2 = input2:double()
				input3 = input3:double()
				target = target:double()
		     elseif opt.type == 'cuda' then
				input1 = input1:cuda()
				input2 = input2:cuda()
				input3 = input3:cuda()
				target = target:cuda()
			 end
		     table.insert(inputs1, input1)
		     table.insert(inputs2, input2)
		     table.insert(inputs3, input3)
		     table.insert(targets, target)
		end
      end

	  if #inputs1~=0 then
		  -- Get parameters
		  if not parameters then
			  if opt.type == 'cuda' then
				parameters = torch.CudaTensor(num_parameters)
			  else
				parameters = torch.Tensor(num_parameters)
			  end
		  end
		  parameters[{{1, num_bank1}}] = parameters_bank1
		  parameters[{{num_bank1+1, num_bank1+num_bank2}}] = parameters_bank2
		  parameters[{{num_bank1+num_bank2+1, num_bank1+num_bank2+num_bank3}}] = parameters_bank3
		  parameters[{{num_bank1+num_bank2+num_bank3+1, num_parameters}}] = parameters_linear

		  -- create closure to evaluate f(X) and df/dX
		  local feval = function(x)
		                   -- get new parameters
		                   if x ~= parameters then
		                      parameters:copy(x)
		                      parameters_bank1[{{}}] = parameters[{{1, num_bank1}}]
		                      parameters_bank2[{{}}] = parameters[{{num_bank1+1, num_bank1+num_bank2}}]
		                      parameters_bank3[{{}}] = parameters[{{num_bank1+num_bank2+1, num_bank1+num_bank2+num_bank3}}]
		                      parameters_linear[{{}}] = parameters[{{num_bank1+num_bank2+num_bank3+1, num_parameters}}]
		                   end

		                   -- reset gradients
		                   gradParameters_bank1:zero()
		                   gradParameters_bank2:zero()
		                   gradParameters_bank3:zero()
		                   gradParameters_linear:zero()

		                   -- f is the average of all criterions
		                   local f = 0

		                   -- evaluate function for complete mini batch
		                   for i = 1,#inputs1 do
		                      -- estimate f
							  local output_conv = {}
							  if opt.type == 'cuda' then
							  		output_conv = torch.CudaTensor( 3, nstates[2], 9, 9 )
							  else
							  		output_conv = torch.Tensor( 3, nstates[2], 9, 9 )
							  end
							  output_conv[{1}] = bank1:forward(inputs1[i])
							  output_conv[{2}] = bank2:forward(inputs2[i])
							  output_conv[{3}] = bank3:forward(inputs3[i])
							  local output_nn = linear_nn:forward(output_conv)
							  local err = criterion:forward(output_nn, targets[i])
							  f = f + err

							  -- estimate df/dW
							  local df_do = criterion:backward(output_nn, targets[i])
							  df_do_nn = linear_nn:backward(output_conv, df_do)
							  bank1:backward(inputs1[i], df_do_nn)
							  bank2:backward(inputs2[i], df_do_nn)
							  bank3:backward(inputs3[i], df_do_nn)

		                      -- update mse
							  mean_square_error = mean_square_error + err
		                      --confusion:add(output, targets[i])
		                   end

		                   -- normalize gradients and f(X)
		                   gradParameters_bank1:div(#inputs1)
		                   gradParameters_bank2:div(#inputs1)
		                   gradParameters_bank3:div(#inputs1)
		                   gradParameters_linear:div(#inputs1)
		                   f = f/#inputs1

		                   -- update gradParameters
		                   if not gradParameters then
				               if opt.type == 'cuda' then
				               		gradParameters = torch.CudaTensor(num_parameters)
				               else
				               		gradParameters = torch.Tensor(num_parameters)
				               end
		                   end
		                   gradParameters[{{1, num_bank1}}] = gradParameters_bank1
		                   gradParameters[{{num_bank1+1, num_bank1+num_bank2}}] = gradParameters_bank2
		                   gradParameters[{{num_bank1+num_bank2+1,num_bank1+num_bank2+num_bank3}}] = gradParameters_bank3
		                   gradParameters[{{num_bank1+num_bank2+num_bank3+1,num_parameters}}] = gradParameters_linear

		                   -- return f and df/dX
		                   return f,gradParameters
		                end

		  -- optimize on current mini-batch
		  if optimMethod == optim.asgd then
		     _,_,average = optimMethod(feval, parameters, optimState)
		  else
		     optimMethod(feval, parameters, optimState)
		  end

	   -- update parameters
	   parameters_bank1[{{}}] = parameters[{{1, num_bank1}}]
	   parameters_bank2[{{}}] = parameters[{{num_bank1+1, num_bank1+num_bank2}}]
	   parameters_bank3[{{}}] = parameters[{{num_bank1+num_bank2+1, num_bank1+num_bank2+num_bank3}}]
	   parameters_linear[{{}}] = parameters[{{num_bank1+num_bank2+num_bank3+1, num_parameters}}]
	end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print mse
   mean_square_error = mean_square_error/train_counter
   print("mean-square error of 1 sample: "..(mean_square_error)..", train_counter="..train_counter)

   -- update logger/plot
   trainLogger:add{['mean-square error (train set)'] = mean_square_error}
   if opt.plot then
      trainLogger:style{['mean-square error (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'bank1.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving bank1 to '..filename)
   torch.save(filename, bank1)

   filename = paths.concat(opt.save, 'bank2.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving bank2 to '..filename)
   torch.save(filename, bank2)

   filename = paths.concat(opt.save, 'bank3.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving bank3 to '..filename)
   torch.save(filename, bank3)

   filename = paths.concat(opt.save, 'linear_nn.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving linear_nn to '..filename)
   torch.save(filename, linear_nn)

   -- next epoch
   --confusion:zero()
   mean_square_error = 0.0
   epoch = epoch + 1
end
