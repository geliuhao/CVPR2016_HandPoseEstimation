----------------------------------------------------------------------
-- 5. testing
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
      parameters_bank1[{{}}] = parameters[{{1, num_bank1}}]
      parameters_bank2[{{}}] = parameters[{{num_bank1+1, num_bank1+num_bank2}}]
      parameters_bank3[{{}}] = parameters[{{num_bank1+num_bank2+1, num_bank1+num_bank2+num_bank3}}]
      parameters_linear[{{}}] = parameters[{{num_bank1+num_bank2+num_bank3+1, num_parameters}}]
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   --model:evaluate()
   bank1:evaluate()
   bank2:evaluate()
   bank3:evaluate()
   linear_nn:evaluate()

   -- test over test data
   test_counter = 0
   print('==> testing on test set:')
   for t = 1,tesize do
      -- disp progress
      xlua.progress(t, tesize)
	  
	  if teValid[{t,1}]~=0 then
		  test_counter = test_counter + 1
		  -- get new sample
		  --local input = testData.data[t]
		  local input1 = test_im_96[{{t}}]
		  local input2 = test_im_48[{{t}}]
		  local input3 = test_im_24[{{t}}]

		  --local target = testData.labels[t]
		  local target = nn.Reshape(noutputs):forward(telabels[t])

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

		  -- test sample
		  --local pred = model:forward(input)
		  local output_conv = {}
		  if opt.type == 'cuda' then
		  		output_conv = torch.CudaTensor( 3, nstates[2], 9, 9 )
		  else
		  		output_conv = torch.Tensor( 3, nstates[2], 9, 9 )
		  end
		  output_conv[{1}] = bank1:forward(input1)
		  output_conv[{2}] = bank2:forward(input2)
		  output_conv[{3}] = bank3:forward(input3)
		  local output_nn = linear_nn:forward(output_conv)
		  local err = criterion:forward(output_nn, target)

		  --confusion:add(pred, target)
		  mean_square_error = mean_square_error + err
	end
   end

   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print mse
   mean_square_error = mean_square_error/test_counter
   print("mean-square error of 1 sample: "..(mean_square_error)..", test_counter="..test_counter)

   -- update log/plot
   testLogger:add{['mean-square error (test set)'] = mean_square_error}
   if opt.plot then
      testLogger:style{['mean-square error (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
      parameters_bank1[{{}}] = parameters[{{1, num_bank1}}]
      parameters_bank2[{{}}] = parameters[{{num_bank1+1, num_bank1+num_bank2}}]
      parameters_bank3[{{}}] = parameters[{{num_bank1+num_bank2+1, num_bank1+num_bank2+num_bank3}}]
      parameters_linear[{{}}] = parameters[{{num_bank1+num_bank2+num_bank3+1, num_parameters}}]
   end
   
   -- next iteration:
   --confusion:zero()
   mean_square_error = 0.0
end
