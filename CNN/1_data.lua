----------------------------------------------------------------------
-- 1. load data
-- This code is used for training OBB XY view, if you want to train YZ/ZX view, just replacing OBB_XY with OBB_YZ/OBB_ZX
-- Created by Liuhao Ge on 08/08/2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
matio = require 'matio'
matio.use_lua_strings = true

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-test_index', 1, 'cross validation')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> loading images'

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.
----------------------------------------------------------------------
-- training/test size

if opt.size == 'full' then
    print '==> using regular, full training data'
	SUBJECT_NUM = 9
	GESTURE_NUM = 17
elseif opt.size == 'small' then
    print '==> using reduced training data, for fast experiments'
	SUBJECT_NUM = 9
	GESTURE_NUM = 1	--8
end

--trsize = 72757
--tesize = 8252
data_dir = "../cvpr15_MSRAHandGestureDB"

nheatmaps = 21
JOINT_NUM = 21
INPUT_SZ = 96

subject_names = { "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8" }
gesture_names = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "I", "IP", "L", "MP", "RP", "T", "TIP", "Y" }
--valid_joints = { 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1 }
valid_joints = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
----------------------------------------------------------------------
-- 1 Load data
-- 1.1 Load the image and ground truth data
local table_frame_num = {}
local table_im_96 = {}
local table_jnt_uv= {}
local table_valid = {}

for i_subject = 1,SUBJECT_NUM do
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

	local subject_im_96 = torch.Tensor(frame_num, INPUT_SZ, INPUT_SZ)
	local subject_valid = torch.Tensor(frame_num, 1)
	local subject_jnt_uv = torch.Tensor(frame_num, nheatmaps, 2)
	start_index = 0
	end_index = 0
	for i_gesture = 1,GESTURE_NUM do
		start_index = end_index + 1
		end_index = end_index + (#table_im_curr[i_gesture])[1]

		subject_im_96[{{start_index, end_index}}] = table_im_curr[i_gesture]
		subject_valid[{{start_index, end_index}}] = table_valid_curr[i_gesture]
		subject_jnt_uv[{{start_index, end_index}}] = table_jnt_curr[i_gesture]
	end
	table.insert(table_frame_num, frame_num)
	table.insert(table_im_96, subject_im_96)
	table.insert(table_valid, subject_valid)
	table.insert(table_jnt_uv, subject_jnt_uv)
end

-- 1.2 Generate training and testing data
trsize=0
tesize=0
for i_subject = 1,SUBJECT_NUM do
	if i_subject == opt.test_index then
		tesize = tesize + table_frame_num[i_subject]
	else
		trsize = trsize + table_frame_num[i_subject]
	end
end

train_im_96 = torch.Tensor(trsize,INPUT_SZ, INPUT_SZ)
trValid = torch.Tensor(trsize,1)
train_jnt_uv = torch.Tensor(trsize,nheatmaps,2)

test_im_96 = torch.Tensor(tesize,INPUT_SZ, INPUT_SZ)
teValid = torch.Tensor(tesize,1)
test_jnt_uv = torch.Tensor(tesize,nheatmaps,2)

train_start_index = 0
train_end_index = 0
test_start_index = 0
test_end_index = 0
for i_subject = 1,SUBJECT_NUM do
	if i_subject == opt.test_index then
		test_start_index = test_end_index + 1
		test_end_index = test_end_index + table_frame_num[i_subject]
		test_im_96[{{test_start_index, test_end_index}}] = table_im_96[i_subject]
		teValid[{{test_start_index, test_end_index}}] = table_valid[i_subject]
		test_jnt_uv[{{test_start_index, test_end_index}}] = table_jnt_uv[i_subject]
	else
		train_start_index = train_end_index + 1
		train_end_index = train_end_index + table_frame_num[i_subject]
		train_im_96[{{train_start_index, train_end_index}}] = table_im_96[i_subject]
		trValid[{{train_start_index, train_end_index}}] = table_valid[i_subject]
		train_jnt_uv[{{train_start_index, train_end_index}}] = table_jnt_uv[i_subject]
	end
end

----------------------------------------------------------------------
-- 2 Local Contrastive Normalization
show_sample_num = 6
-- 2.1 Local Contrastive Normalization for training data
--train_im_48 = image.scale(train_im_96, 48, 48)
--train_im_24 = image.scale(train_im_48, 24, 24)

neighborhood = image.gaussian1D(9,0.8,1,true)
--neighborhood = image.gaussian1D(23)
threshold = 1	--0.0001
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, threshold):float()

--neighborhood = torch.Tensor(31,31)
--normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 0.0001):float()

for i = 1,trsize do
    train_im_96[{ {i},{},{} }] = normalization:forward(train_im_96[{ {i},{},{} }])
--    train_im_48[{ {i},{},{} }] = normalization:forward(train_im_48[{ {i},{},{} }])
--    train_im_24[{ {i},{},{} }] = normalization:forward(train_im_24[{ {i},{},{} }])
end
--
--train_im_48 = image.gaussianpyramid(train_images.images_96, {0.5})[1]
--train_im_24 = image.gaussianpyramid(train_im_48, {0.5})[1]

train_im_48 = image.scale(train_im_96, 48, 48)
train_im_24 = image.scale(train_im_48, 24, 24)

if itorch then
	show_images = train_im_96[{{1,show_sample_num}}]
	itorch.image(show_images)
	show_images = train_im_48[{{1,show_sample_num}}]
	itorch.image(show_images)
	show_images = train_im_24[{{1,show_sample_num}}]
	itorch.image(show_images)
end

-- 2.2 Local Contrastive Normalization for testing data
--test_im_48 = image.scale(test_im_96, 48, 48)
--test_im_24 = image.scale(test_im_48, 24, 24)

for i = 1,tesize do
    test_im_96[{ {i},{},{} }] = normalization:forward(test_im_96[{ {i},{},{} }])
--    test_im_48[{ {i},{},{} }] = normalization:forward(test_im_48[{ {i},{},{} }])
--    test_im_24[{ {i},{},{} }] = normalization:forward(test_im_24[{ {i},{},{} }])
end
--
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

----------------------------------------------------------------------
-- 3 Generate Heat Maps
heatmap_sz = 18
noutputs = nheatmaps*heatmap_sz*heatmap_sz

gaussian_sigma = 0.1	--0.1 效果最好 --0.07 跟原文Fig.5最像 --0.05

--[[
trValid = torch.Tensor(trsize,1)
trValid:fill(1)
teValid = torch.Tensor(tesize,1)
teValid:fill(1)
--]]
	  -- 3.1 convert training labels:
      trlabels = torch.Tensor( trsize, nheatmaps, heatmap_sz, heatmap_sz )
      for i = 1,trsize do
		 curr_labels = torch.Tensor( nheatmaps, heatmap_sz, heatmap_sz )
		 curr_labels:fill(0)
		 for j = 1,nheatmaps do
			--curr_labels[{j, train_jnt_uv[{i, j, 2}], train_jnt_uv[{i, j, 1}]}] = 1.0
			mean_horz = train_jnt_uv[{i, j, 1}]/heatmap_sz
			mean_vert = train_jnt_uv[{i, j, 2}]/heatmap_sz
			if mean_horz>=1 or mean_vert>=1 or mean_horz<0 or mean_vert<0 then
				trValid[{i,1}] = 0		-- out of limit
				break
			end
			curr_labels[{j}]=image.gaussian(heatmap_sz,gaussian_sigma,1,true,heatmap_sz,heatmap_sz,gaussian_sigma,gaussian_sigma,mean_horz,mean_vert)
		 end
         --trlabels[{ i }] = nn.Reshape(1, noutputs):forward(curr_labels)
		 trlabels[{ i }] = curr_labels
      end

      -- 3.2 convert testing labels
      telabels = torch.Tensor( tesize, nheatmaps, heatmap_sz, heatmap_sz )
      for i = 1,tesize do
		 curr_labels = torch.Tensor( nheatmaps, heatmap_sz, heatmap_sz )
		 curr_labels:fill(0)
		 for j = 1,nheatmaps do
			--curr_labels[{j, test_jnt_uv[{i, j, 2}], test_jnt_uv[{i, j, 1}]}] = 1.0
			mean_horz = test_jnt_uv[{i, j, 1}]/heatmap_sz
			mean_vert = test_jnt_uv[{i, j, 2}]/heatmap_sz
			if mean_horz>=1 or mean_vert>=1 or mean_horz<0 or mean_vert<0 then
				teValid[{i,1}] = 0		-- out of limit
				break
			end
			curr_labels[{j}]=image.gaussian(heatmap_sz,gaussian_sigma,1,true,heatmap_sz,heatmap_sz,gaussian_sigma,gaussian_sigma,mean_horz,mean_vert)
		 end
         --telabels[{ i }] = nn.Reshape(1, noutputs):forward(curr_labels)
		 telabels[{ i }] = curr_labels
      end
