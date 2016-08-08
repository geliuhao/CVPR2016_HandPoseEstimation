----------------------------------------------------------------------
-- 3. define loss function
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Loss Function')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'mse', 'type of loss function to minimize: nll | mse | margin')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

----------------------------------------------------------------------
print '==> define loss'

if opt.loss == 'margin' then

   -- This loss takes a vector of classes, and the index of
   -- the grountruth class as arguments. It is an SVM-like loss
   -- with a default margin of 1.

   criterion = nn.MultiMarginCriterion()

elseif opt.loss == 'nll' then

   -- This loss requires the outputs of the trainable model to
   -- be properly normalized log-probabilities, which can be
   -- achieved using a softmax function

   linear_nn:add(nn.LogSoftMax())

   -- The loss works like the MultiMarginCriterion: it takes
   -- a vector of classes, and the index of the grountruth class
   -- as arguments.

   criterion = nn.ClassNLLCriterion()

elseif opt.loss == 'mse' then

   -- for MSE, we add a tanh, to restrict the model's output
   --linear_nn:add(nn.Tanh())
   linear_nn:add(nn.ReLU())
   --linear_nn:add(nn.Sigmoid())

   -- The mean-square error is not recommended for classification
   -- tasks, as it typically tries to do too much, by exactly modeling
   -- the 1-of-N distribution. For the sake of showing more examples,
   -- we still provide it here:

   criterion = nn.MSECriterion()
   criterion.sizeAverage = false

   -- Compared to the other losses, the MSE criterion needs a distribution
   -- as a target, instead of an index. Indeed, it is a regression loss!
   -- So we need to transform the entire label vectors:

   --if trainData then
      -- move to 1_data.lua
   --end

else

   error('unknown -loss')

end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
