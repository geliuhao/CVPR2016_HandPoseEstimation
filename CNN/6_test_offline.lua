----------------------------------------------------------------------
-- 6. after some iterations, test on testing set and generate heat-maps
-- This code is used for generating OBB XY view's heat-map, if you want to generate YZ/ZX view's heat-map, just replacing OBB_XY with OBB_YZ/OBB_ZX
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'mattorch'

matio = require 'matio'
matio.use_lua_strings = true

if not opt then
	print '==> offline testing'

	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('hand pose recovery')
	cmd:text()
	cmd:text('Options:')
	-- global:
	cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
	cmd:option('-threads', 2, 'number of threads')
	-- data:
	cmd:option('-size', 'full', 'how many samples do we load: small | full')
	cmd:option('-test_index', 1, 'cross validation')
	cmd:option('-experiment_index', 0, 'experiment index')
	-- model:
	cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
	-- loss:
	cmd:option('-loss', 'mse', 'type of loss function to minimize: nll | mse | margin')
	-- training:
	cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
	cmd:option('-plot', false, 'live plot')
	cmd:option('-type', 'cuda', 'type: double | float | cuda')
	cmd:text()
	opt = cmd:parse(arg or {})
end

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

bank1 = torch.load(opt.save..'/bank1.net')
bank2 = torch.load(opt.save..'/bank2.net')
bank3 = torch.load(opt.save..'/bank3.net')
linear_nn = torch.load(opt.save..'/linear_nn.net')

-----------------------------------------------------------------------
print '==> loading test image data'

if opt.size == 'full' then
    print '==> using regular, full training data'
	SUBJECT_NUM = 9
	GESTURE_NUM = 17
elseif opt.size == 'small' then
    print '==> using reduced training data, for fast experiments'
	SUBJECT_NUM = 9
	GESTURE_NUM = 1
end

data_dir = "../cvpr15_MSRAHandGestureDB"

nheatmaps = 21
JOINT_NUM = 21
INPUT_SZ = 96

subject_names = { "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8" }
gesture_names = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "I", "IP", "L", "MP", "RP", "T", "TIP", "Y" }
valid_joints = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
----------------------------------------------------------------------
-- 1 Load data
-- 1.1 Load the image and ground truth data
	i_subject = opt.test_index
	local frame_num = 0
	local table_im_curr = {}
	local table_valid_curr = {}
	local table_jnt_curr = {}
	for i_gesture = 1,GESTURE_NUM do
		data_path = data_dir.."/"..subject_names[i_subject].."/"..gesture_names[i_gesture].."/"

		local images = matio.load(data_path.."OBB_XY.mat")
		table.insert(table_im_curr, images.OBB_XY)

		local cur_valid = matio.load(data_path.."valid.mat")
		table.insert(table_valid_curr, cur_valid.valid)

		local joint_data = matio.load(data_path.."OBB_XY_heatmaps.mat")
		local gesture_jnt_uv = torch.Tensor((#images.OBB_XY)[1], nheatmaps, 2)
		local i_heat = 0
		for i_joint = 1,JOINT_NUM do
			if valid_joints[i_joint]==1 then
				i_heat = i_heat + 1
				gesture_jnt_uv[{{},{i_heat},{}}] = joint_data.OBB_XY_heatmaps[{{}, {i_joint}, {}}]
			end
		end
		table.insert(table_jnt_curr, gesture_jnt_uv)

		frame_num = frame_num + (#images.OBB_XY)[1]
	end

-- 1.2 Generate testing data
	test_im_96 = torch.Tensor(frame_num, INPUT_SZ, INPUT_SZ)
	teValid = torch.Tensor(frame_num,1)
	test_jnt_uv = torch.Tensor(frame_num, nheatmaps, 2)
	start_index = 0
	end_index = 0
	for i_gesture = 1,GESTURE_NUM do
		start_index = end_index + 1
		end_index = end_index + (#table_im_curr[i_gesture])[1]

		test_im_96[{{start_index, end_index}}] = table_im_curr[i_gesture]
		teValid[{{start_index, end_index}}] = table_valid_curr[i_gesture]
		test_jnt_uv[{{start_index, end_index}}] = table_jnt_curr[i_gesture]
	end

----------------------------------------------------------------------
-- 2 Local Contrastive Normalization
show_sample_num = 6
tesize = frame_num

neighborhood = image.gaussian1D(9,0.8,1,true)
--neighborhood = image.gaussian1D(23)
threshold = 1	--0.0001
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, threshold):float()

-- 2.1 Load the test image data.

for i = 1,tesize do
    test_im_96[{ {i},{},{} }] = normalization:forward(test_im_96[{ {i},{},{} }])
    --test_im_48[{ {i},{},{} }] = normalization:forward(test_im_48[{ {i},{},{} }])
    --test_im_24[{ {i},{},{} }] = normalization:forward(test_im_24[{ {i},{},{} }])
end

--test_im_48 = image.gaussianpyramid(test_images.images_96, {0.5})[1]
--test_im_24 = image.gaussianpyramid(test_im_48, {0.5})[1]
test_im_48 = image.scale(test_im_96, 48, 48)
test_im_24 = image.scale(test_im_48, 24, 24)

if itorch then
	show_images = test_im_96[{{1,show_sample_num}}]
	itorch.image(show_images)
	show_images = test_im_48[{{1,show_sample_num}}]
	itorch.image(show_images)
	show_images = test_im_24[{{1,show_sample_num}}]
	itorch.image(show_images)
end

print '==> OK'

nheatmaps = 21
heatmap_sz = 18
noutputs = nheatmaps*heatmap_sz*heatmap_sz

gaussian_sigma = 0.1	--0.05

--teValid = torch.Tensor(tesize,1)
--teValid:fill(1)

-- 3.2 convert testing labels
telabels = torch.Tensor( tesize, nheatmaps, heatmap_sz, heatmap_sz )
teInvalidNum = 0
for i = 1,tesize do
	curr_labels = torch.Tensor( nheatmaps, heatmap_sz, heatmap_sz )
	curr_labels:fill(0)
	for j = 1,nheatmaps do
			--curr_labels[{j, test_jnt_uv[{i, j, 2}], test_jnt_uv[{i, j, 1}]}] = 1.0
			mean_horz = test_jnt_uv[{i, j, 1}]/heatmap_sz
			mean_vert = test_jnt_uv[{i, j, 2}]/heatmap_sz
			if mean_horz>=1 or mean_vert>=1 or mean_horz<0 or mean_vert<0 then
				teValid[{i,1}] = 0		-- out of limit
				teInvalidNum = teInvalidNum + 1
				break
			end
			curr_labels[{j}]=image.gaussian(heatmap_sz,gaussian_sigma,1,true,heatmap_sz,heatmap_sz,gaussian_sigma,gaussian_sigma,mean_horz,mean_vert)
	end
	--telabels[{ i }] = nn.Reshape(1, noutputs):forward(curr_labels)
	telabels[{ i }] = curr_labels
end
print '==> OK'
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
-- 4. Testing
nstates = {16,32,6804}
linear_input_num = 7776

criterion = nn.MSECriterion()
criterion.sizeAverage = false

local time = sys.clock()

-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
--model:evaluate()
bank1:evaluate()
bank2:evaluate()
bank3:evaluate()
linear_nn:evaluate()

--output_heatmaps = torch.Tensor( tesize-teInvalidNum, nheatmaps, heatmap_sz, heatmap_sz )
output_heatmaps = torch.Tensor( tesize, nheatmaps, heatmap_sz, heatmap_sz )

print '==> Testing!'

err = 0.0
test_counter = 0
for t=1,tesize do
	xlua.progress(t, tesize)
	
	if teValid[{t,1}]~=0 then
		test_counter = test_counter + 1

		input1 = test_im_96[{{t}}]
		input2 = test_im_48[{{t}}]
		input3 = test_im_24[{{t}}]

		target = nn.Reshape(noutputs):forward(telabels[t])

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

		output_conv = {}
		if opt.type == 'cuda' then
			output_conv = torch.CudaTensor( 3, nstates[2], 9, 9 )
		else
			output_conv = torch.Tensor( 3, nstates[2], 9, 9 )
		end
		output_conv[{1}] = bank1:forward(input1)	-- 0.0009s
		output_conv[{2}] = bank2:forward(input2)	-- 0.0006s
		output_conv[{3}] = bank3:forward(input3)	-- 0.0004s
		output_nn = linear_nn:forward(output_conv)	-- 0.0008s
		err = err + criterion:forward(output_nn, target)

		--output_heatmaps[{test_counter}] = nn.Reshape(nheatmaps, heatmap_sz, heatmap_sz):forward(output_nn:float())
		output_heatmaps[{t}] = nn.Reshape(nheatmaps, heatmap_sz, heatmap_sz):forward(output_nn:float())
		target_heatmaps = nn.Reshape(nheatmaps, heatmap_sz, heatmap_sz):forward(target:float())

		if itorch then
			if (t%500)==1 then
				print("Sample "..t..": output_heatmaps")
				itorch.image(output_heatmaps[{t}])
				print("Sample "..t..": target_heatmaps")
				itorch.image(target_heatmaps)
			end
		end
	end
end

   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print mse
   err = err/test_counter
   print("mean-square error of 1 sample: "..(err))

torch.setdefaulttensortype('torch.DoubleTensor')
mattorch.save(paths.concat(opt.save, opt.experiment_index, 'est_heatmaps_OBB_XY.mat'), output_heatmaps:double())
