// Projection.cpp : Defines the entry point for the console application.
//
//  Created by Liuhao Ge on 08/08/2016.
//

#include "stdafx.h"

#include <string>
#include <stdint.h>
#include <cv.h>
#include <highgui.h>

#include "extern/include/mat.h"
#include "extern/include/matrix.h"
#include "BoundingBox.h"

using namespace std;

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }
#define SAFE_MAT_DELETE(x) if (x != NULL) { mxFree(x); x = NULL; }

#define MAX_KINECTS 1
#define NUM_WORKER_THREADS 6
#define src_width 320
#define src_height 240
#define src_dim (src_width * src_height)

#define DATASET_DIR std::string("..\\..\\cvpr15_MSRAHandGestureDB")
#define OUT_SZ 96
#define HEAT_SZ 18
#define PARAM_SZ 5

#define SUBJECT_NUM 9
#define GESTURE_NUM 17
#define JOINT_NUM 21
#define MAX_IMAGE_NUM 500
#define MAX_DEPTH 2001

const std::string subject_names[SUBJECT_NUM] = { "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8" };

const std::string gesture_names[GESTURE_NUM] = { "1", "2", "3", "4", "5", "6", "7", "8", "9",
"I", "IP", "L", "MP", "RP", "T", "TIP", "Y" };

const std::string AABB_names[3] = { "AABB_XY", "AABB_YZ", "AABB_ZX" };
const std::string OBB_names[3] = { "OBB_XY", "OBB_YZ", "OBB_ZX" };

// Depth Camera Image data
#define nXRes 320
#define nYRes 240
const double fFocal_MSRA_ = 241.42;	// mm

float cur_xyz_data[MAX_KINECTS][src_dim * 3];
float cur_uvd_data[MAX_KINECTS][src_dim * 3];
uint32_t gesture_image_num[GESTURE_NUM];	// number of images for each gesture
uint32_t cur_image = 0;
uint8_t cur_gesture = 0;
uint8_t cur_subject = 0;
uint32_t cur_pixel_num = 0;
BoundingBox cur_bounding_box_OBB;
float ground_truth_xyz[MAX_IMAGE_NUM][JOINT_NUM][3];
float cur_xyz_ground_truth[MAX_KINECTS][JOINT_NUM * 3];

double *trans_params = NULL;	// 500 x 5
int trans_params_dims[2] = { 0 };
float *OBB_XYZ = NULL;
int total_image_num = 0;
uint32_t cur_global_image = 0;

float *OBB_proj[3] = { NULL, NULL, NULL };
double *OBB_heatmap[3] = { NULL, NULL, NULL };
double *OBB_params[3] = { NULL, NULL, NULL };
double *OBB_trans = NULL;
double *OBB_length = NULL;

void convertDepthImageToProjective2(const float* aDepth, const cv::Rect& bounding_box, float* aProjective)
{
	int32_t nIndex = 0;
	for (int i = 0; i < bounding_box.height; i++)
	{
		for (int j = 0; j < bounding_box.width; j++)
		{
			//float depth = aDepth[bounding_box.width * i + j];
			aProjective[nIndex * 3] = static_cast<float>(j + bounding_box.x);
			aProjective[nIndex * 3 + 1] = static_cast<float>(i + bounding_box.y);
			aProjective[nIndex * 3 + 2] = aDepth[nIndex];
			++nIndex;
		}
	}
}

void convertDepthToWorldCoordinates2(const float* uvd, float* xyz, const uint32_t nCount)
{
	for (int32_t i = 0; i < (int32_t)nCount; i++) {
		xyz[i * 3] = -(float(nXRes)/2 - uvd[i * 3]) * uvd[i * 3 + 2] / fFocal_MSRA_;
		xyz[i * 3 + 1] = -(uvd[i * 3 + 1] - float(nYRes) / 2) * uvd[i * 3 + 2] / fFocal_MSRA_;
		xyz[i * 3 + 2] = uvd[i * 3 + 2];
	}
}

void loadCurrentSubject()
{
	total_image_num = 0;
	for (int i_gesture = 0; i_gesture < GESTURE_NUM; ++i_gesture)
	{
		char joint_path[255];
		sprintf(joint_path, "%s\\%s\\%s\\joint.txt",
			DATASET_DIR.c_str(), subject_names[cur_subject].c_str(), gesture_names[i_gesture].c_str());
		FILE *pJointFile = fopen(joint_path, "r");
		if (!pJointFile)
		{
			std::cout << "Could not open joint.txt file" << endl;
			exit(EXIT_FAILURE);
		}
		fscanf(pJointFile, "%d\n", &gesture_image_num[i_gesture]);
		total_image_num += gesture_image_num[i_gesture];
		fclose(pJointFile);
	}
	SAFE_DELETE_ARR(OBB_XYZ);
	OBB_XYZ = new float[total_image_num * JOINT_NUM * 3];
}

void loadCurrentGesture(bool print_to_screen = true)
{
	if (print_to_screen)
	{
		cout << "loading gesture: " << subject_names[cur_subject] << "/" << gesture_names[cur_gesture] << endl;
	}
	// 1. Read ground truth
	char joint_path[255];
	sprintf(joint_path, "%s\\%s\\%s\\joint.txt",
		DATASET_DIR.c_str(), subject_names[cur_subject].c_str(), gesture_names[cur_gesture].c_str());
	FILE *pJointFile = fopen(joint_path, "r");
	if (!pJointFile)
	{
		cout << "Could not open joint.txt file" << endl;
		exit(EXIT_FAILURE);
	}
	int cur_image_num = 0;
	fscanf(pJointFile, "%d\n", &cur_image_num);

	for (int i_image = 0; i_image < cur_image_num; ++i_image)
	{
		for (int i_joint = 0; i_joint < JOINT_NUM; ++i_joint)
		{
			int tmp = i_image * JOINT_NUM * 3 + i_joint * 3;
			fscanf(pJointFile, "%f %f %f", &ground_truth_xyz[i_image][i_joint][0],
				&ground_truth_xyz[i_image][i_joint][1], &ground_truth_xyz[i_image][i_joint][2]);
			ground_truth_xyz[i_image][i_joint][2] *= (-1.0);
			if (ground_truth_xyz[i_image][i_joint][2] < 0)
			{
				cout << "==> Negative value of ground truth depth!" << endl;
			}
			if (i_joint < JOINT_NUM - 1)
			{
				fscanf(pJointFile, " ");
			}
			else
			{
				fscanf(pJointFile, "\n");
			}
		}
	}
	fclose(pJointFile);

	// 2. reallocate memory
	for (int i = 0; i < 3; ++i)
	{
		SAFE_DELETE_ARR(OBB_proj[i]);
		OBB_proj[i] = new float[cur_image_num * OUT_SZ * OUT_SZ];

		SAFE_DELETE_ARR(OBB_heatmap[i]);
		OBB_heatmap[i] = new double[cur_image_num * JOINT_NUM * 2];

		SAFE_DELETE_ARR(OBB_params[i]);
		OBB_params[i] = new double[cur_image_num * PARAM_SZ];

		SAFE_DELETE_ARR(OBB_trans);
		OBB_trans = new double[cur_image_num * 4 * 4];

		SAFE_DELETE_ARR(OBB_length);
		OBB_length = new double[cur_image_num * 3];
	}
}

void loadCurrentImage(bool print_to_screen = true) {
	if (print_to_screen)
	{
		cout << "loading image: " << subject_names[cur_subject] << "/" << gesture_names[cur_gesture] << "/" << cur_image << endl;
	}

	// 1. read bounding box, depth data
	cv::Rect bounding_box;
	char depth_path[255];
	sprintf(depth_path, "%s\\%s\\%s\\%06d_depth.bin", DATASET_DIR.c_str(), subject_names[cur_subject].c_str(), gesture_names[cur_gesture].c_str(), cur_image);
	FILE *pDepthFile = fopen(depth_path, "rb");
	if (!pDepthFile)
	{
		std::cout << "Could not open depth.bin file" << endl;
		exit(EXIT_FAILURE);
	}
	int img_width = 0, img_height = 0, right = 0, bottom = 0;
	fread(&img_width, sizeof(int), 1, pDepthFile);
	fread(&img_height, sizeof(int), 1, pDepthFile);
	if (img_width != src_width || img_height != src_height)
	{
		std::cout << "Width or height does not match" << endl;
		exit(EXIT_FAILURE);
	}
	fread(&bounding_box.x, sizeof(int), 1, pDepthFile);
	fread(&bounding_box.y, sizeof(int), 1, pDepthFile);
	fread(&right, sizeof(int), 1, pDepthFile);
	fread(&bottom, sizeof(int), 1, pDepthFile);
	bounding_box.width = right - bounding_box.x;
	bounding_box.height = bottom - bounding_box.y;

	cur_pixel_num = bounding_box.width * bounding_box.height;
	float* pDepth = new float[cur_pixel_num];
	fread(pDepth, sizeof(float), cur_pixel_num, pDepthFile);
	fclose(pDepthFile);

	convertDepthImageToProjective2(pDepth, bounding_box, cur_uvd_data[0]);
	convertDepthToWorldCoordinates2(cur_uvd_data[0], cur_xyz_data[0], cur_pixel_num);

	//if (!valid_images[cur_image])
	//{
	//	cout << "=> Current Image is Invalid!" << endl;
	//	return;
	//}

	//2. get ground truth
	for (int i = 0; i < JOINT_NUM; ++i)
	{
		cur_xyz_ground_truth[0][3 * i] = ground_truth_xyz[cur_image][i][0];
		cur_xyz_ground_truth[0][3 * i + 1] = ground_truth_xyz[cur_image][i][1];
		cur_xyz_ground_truth[0][3 * i + 2] = ground_truth_xyz[cur_image][i][2];
	}
	SAFE_DELETE_ARR(pDepth);

	// 3. create the bounding box
	cur_bounding_box_OBB.create_box_OBB(cur_xyz_data[0], cur_pixel_num, cur_xyz_ground_truth[0], JOINT_NUM);

	// 4. project to x-y, y-z, z-x plane
	cv::Mat proj_im_OBB[3];
	cv::Point2f proj_uv_OBB[3][JOINT_NUM];
	cur_bounding_box_OBB.project_direct(proj_im_OBB, proj_uv_OBB, OUT_SZ);

	// 5. save 96*96 images
	int cur_image_num = gesture_image_num[cur_gesture];
	for (int i = 0; i < OUT_SZ; ++i)
	{
		for (int j = 0; j < OUT_SZ; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				OBB_proj[k][j*OUT_SZ*cur_image_num + i*cur_image_num + cur_image] = proj_im_OBB[k].at<float>(i, j);
			}
		}
	}

	// 6. save transform matrix & parameters
	Float4x4& OBB_relative_trans = cur_bounding_box_OBB.get_relative_trans();
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			OBB_trans[j * 4 * cur_image_num + i*cur_image_num + cur_image] = OBB_relative_trans(i, j);
		}
	}

	OBB_length[0 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_x_length();
	OBB_length[1 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_y_length();
	OBB_length[2 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_z_length();

	for (int i = 0; i < 3; ++i)
	{
		OBB_params[i][0 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_proj_bounding_box()[i].x;
		OBB_params[i][1 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_proj_bounding_box()[i].y;
		OBB_params[i][2 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_proj_k()[i];
		OBB_params[i][3 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_proj_bounding_box()[i].width;
		OBB_params[i][4 * cur_image_num + cur_image] = cur_bounding_box_OBB.get_proj_bounding_box()[i].height;
	}

	// 7. save ground truth heat-maps for training
	for (int i = 0; i < JOINT_NUM; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			OBB_heatmap[j][0 + i*cur_image_num + cur_image] = proj_uv_OBB[j][i].x * HEAT_SZ / OUT_SZ;
			OBB_heatmap[j][JOINT_NUM*cur_image_num + i*cur_image_num + cur_image] = proj_uv_OBB[j][i].y * HEAT_SZ / OUT_SZ;
			circle(proj_im_OBB[j], proj_uv_OBB[j][i], 3, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		}
	}
	// 8. display projection images
	for (int i = 0; i < 3; ++i)
	{
		cv::namedWindow(OBB_names[i]);
		cv::imshow(OBB_names[i], proj_im_OBB[i]);
	}
}

void save_matlab_file(int dim_num, const mwSize* dims, int array_sz, void* pArray,
	const char* file_name, const char* var_name, mxClassID class_id)
{
	mxArray *pMxArray = mxCreateNumericArray(dim_num, dims, class_id, mxREAL);
	memcpy((void*)(mxGetPr(pMxArray)), pArray, array_sz);

	MATFile *pMatFile = matOpen(file_name, "w");
	matPutVariable(pMatFile, var_name, pMxArray);
	matClose(pMatFile);
}

void save_gesture_data()
{
	int image_num = gesture_image_num[cur_gesture];
	int sz_im96_array = image_num*OUT_SZ * OUT_SZ * sizeof(OBB_proj[0][0]);
	mwSize im96_dims[3] = { image_num, OUT_SZ, OUT_SZ };
	int sz_heatmap_array = image_num*JOINT_NUM * 2 * sizeof(OBB_heatmap[0][0]);
	mwSize heat_dims[3] = { image_num, JOINT_NUM, 2 };
	int sz_params_array = image_num * PARAM_SZ * sizeof(OBB_params[0][0]);
	mwSize params_dims[2] = { image_num, PARAM_SZ };

	char path[255];
	sprintf(path, "%s\\%s\\%s", DATASET_DIR.c_str(), subject_names[cur_subject].c_str(),
		gesture_names[cur_gesture].c_str());
	char file_name[255];
	char var_name[255];

	for (int i = 0; i < 3; ++i)
	{
		// 1. Save image_num x 96x96 images for OBB
		sprintf(file_name, "%s\\%s.mat", path, OBB_names[i].c_str());
		save_matlab_file(3, im96_dims, sz_im96_array, (void*)OBB_proj[i],
			file_name, OBB_names[i].c_str(), mxSINGLE_CLASS);

		// 2. Save frame_num x 14 x 2 heat-maps for OBB
		sprintf(file_name, "%s\\%s_heatmaps.mat", path, OBB_names[i].c_str());
		sprintf(var_name, "%s_heatmaps", OBB_names[i].c_str());
		save_matlab_file(3, heat_dims, sz_heatmap_array, (void*)OBB_heatmap[i],
			file_name, var_name, mxDOUBLE_CLASS);

		// 3. Save frame_num x PARAM_SZ OBB_params for OBB
		sprintf(file_name, "%s\\%s_params.mat", path, OBB_names[i].c_str());
		sprintf(var_name, "%s_params", OBB_names[i].c_str());
		save_matlab_file(2, params_dims, sz_params_array, (void*)OBB_params[i],
			file_name, var_name, mxDOUBLE_CLASS);
	}

	// 4. Save frame_num x 4 x 4 trans
	int sz_trans_array = image_num * 4 * 4 * sizeof(OBB_trans[0]);
	mwSize trans_dims[3] = { image_num, 4, 4 };

	sprintf(file_name, "%s\\OBB_trans.mat", path);
	save_matlab_file(3, trans_dims, sz_trans_array, (void*)OBB_trans,
		file_name, "OBB_trans", mxDOUBLE_CLASS);

	// 5. Save frame_num x 3 length
	int sz_length_array = image_num * 3 * sizeof(OBB_length[0]);
	mwSize length_dims[2] = { image_num, 3 };

	sprintf(file_name, "%s\\OBB_length.mat", path);
	save_matlab_file(2, length_dims, sz_length_array, (void*)OBB_length,
		file_name, "OBB_length", mxDOUBLE_CLASS);
}

void image_step_forward()
{
	++cur_global_image;
	++cur_image;
	if (cur_image >= gesture_image_num[cur_gesture])
	{
		save_gesture_data();

		cur_image = 0;
		++cur_gesture;
		if (cur_gesture >= GESTURE_NUM)
		{
			cur_gesture = 0;
			++cur_subject;
			return;
		}
		loadCurrentGesture();
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	// Load 1st image
	cur_global_image = 0;
	cur_gesture = 0;
	cur_image = 0;
	cur_subject = 0;
	loadCurrentSubject();
	loadCurrentGesture();
	loadCurrentImage();

	while (true)
	{
		if (cur_subject == SUBJECT_NUM - 1 && cur_gesture == GESTURE_NUM - 1 && cur_image == gesture_image_num[cur_gesture])
		{
			break;
		}
		else
		{
			image_step_forward();
			loadCurrentImage();

			cv::waitKey(10);
		}
	}
	return 0;
}

