// Fusion.cpp : Defines the entry point for the console application.
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
#include "Fuser.h"

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

#define HEATMAP_DIR std::string("..\\..\\heatmaps")
#define HEATMAP_VAR_NAME std::string("x")
#define PCA_DIR std::string("..\\..\\PCA")

#define TEST_SUBJECT 0	// start from 0

const std::string subject_names[SUBJECT_NUM] = { "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8" };

const std::string gesture_names[GESTURE_NUM] = { "1", "2", "3", "4", "5", "6", "7", "8", "9",
												"I", "IP", "L", "MP", "RP", "T", "TIP", "Y" };

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
uint8_t cur_subject = TEST_SUBJECT;
uint32_t cur_pixel_num = 0;
uint32_t cur_estimate_index = 0;
uint32_t total_estimate_num = 0;
float ground_truth_xyz[MAX_IMAGE_NUM][JOINT_NUM][3];
float cur_xyz_ground_truth[MAX_KINECTS][JOINT_NUM * 3];
float cur_xyz_estimated[MAX_KINECTS][JOINT_NUM * 3];

float *estimate_error_distance_arr = NULL;
float *estimate_joints_arr = NULL;
float g_avg_distance = 0.0;
int g_avg_distance_counter = 0;
int g_good_estimations = 0;
float estimation_threshold = 30.0;

double *trans_params = NULL;	// 500 x 5
int trans_params_dims[2] = { 0 };
uint32_t cur_global_image = 0;

double *heatmap_xy = NULL;		// 18 x 18 x 21 x #images
double *heatmap_yz = NULL;		// 18 x 18 x 21 x #images
double *heatmap_zx = NULL;		// 18 x 18 x 21 x #images
int heatmap_dims[4] = { 0 };

Fuser cur_fuser;

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
		xyz[i * 3] = -(float(nXRes) / 2 - uvd[i * 3]) * uvd[i * 3 + 2] / fFocal_MSRA_;
		xyz[i * 3 + 1] = -(uvd[i * 3 + 1] - float(nYRes) / 2) * uvd[i * 3 + 2] / fFocal_MSRA_;
		xyz[i * 3 + 2] = uvd[i * 3 + 2];
	}
}

void read_heatmap(double * heatmap_array, int heatmap_index, cv::Mat& heatmap_mat)
{
	heatmap_mat.release();
	heatmap_mat.create(HEAT_SZ, HEAT_SZ, CV_32FC1);
	for (int i = 0; i < HEAT_SZ; ++i)
	{
		for (int j = 0; j < HEAT_SZ; ++j)
		{
			heatmap_mat.at<float>(i, j) = heatmap_array[heatmap_index + i*HEAT_SZ + j];
		}
	}
}

float get_estimate_err_dist(const float* xyz_estimate, const float* xyz_ground_truth, float* distance)
{
	float avg_distance = 0.0;
	Float3 avg_err_vec(0.0, 0.0, 0.0);

	for (int i = 0; i < JOINT_NUM; ++i)
	{
		distance[i] = 0;
		Float3 cur_err_vec;
		for (int j = 0; j < 3; ++j)
		{
			cur_err_vec[j] = xyz_estimate[i * 3 + j] - xyz_ground_truth[i * 3 + j];
			distance[i] += pow(xyz_estimate[i * 3 + j] - xyz_ground_truth[i * 3 + j], 2);
		}
		Float3::add(avg_err_vec, avg_err_vec, cur_err_vec);
		distance[i] = sqrt(distance[i]);
		avg_distance += distance[i];
	}
	Float3::scale(avg_err_vec, 1.0 / JOINT_NUM);

	return avg_distance / JOINT_NUM;
}

void loadCurrentSubject()
{
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
		fclose(pJointFile);
	}
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
}

void loadCurrentImage(bool print_to_screen = true)
{
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
	cur_fuser.bounding_box_3D.create_box_OBB(cur_xyz_data[0], cur_pixel_num, JOINT_NUM);

	// 4. project to x-y, y-z, z-x plane
	cv::Mat proj_im_OBB[3];
	cv::Point2f proj_uv_OBB[3][JOINT_NUM];
	cur_fuser.bounding_box_3D.project_direct(proj_im_OBB, OUT_SZ);

	// 5. fusion
	for (int i = 0; i < 3; ++i)
	{
		cur_fuser.bounding_box_x[i] = cur_fuser.bounding_box_3D.get_proj_bounding_box()[i].x;
		cur_fuser.bounding_box_y[i] = cur_fuser.bounding_box_3D.get_proj_bounding_box()[i].y;
		cur_fuser.proj_k[i] = cur_fuser.bounding_box_3D.get_proj_k()[i];
		cur_fuser.bounding_box_width[i] = cur_fuser.bounding_box_3D.get_proj_bounding_box()[i].width;
		cur_fuser.bounding_box_height[i] = cur_fuser.bounding_box_3D.get_proj_bounding_box()[i].height;
	}

	int heatmap_coef2 = heatmap_dims[0] * heatmap_dims[1];
	int heatmap_coef1 = heatmap_coef2 * heatmap_dims[2];
	for (int i = 0; i < JOINT_NUM; ++i)
	{
		int heatmap_index = cur_estimate_index*heatmap_coef1 + i*heatmap_coef2;
		read_heatmap(heatmap_xy, heatmap_index, cur_fuser.heatmaps_vec[i * 3]);
		read_heatmap(heatmap_yz, heatmap_index, cur_fuser.heatmaps_vec[i * 3 + 1]);
		read_heatmap(heatmap_zx, heatmap_index, cur_fuser.heatmaps_vec[i * 3 + 2]);
	}

	cur_fuser.fuse(cur_xyz_estimated[0]);

	// 6. calculate error distance
	int tmp_index1 = g_avg_distance_counter*(JOINT_NUM + 1);
	int tmp_index2 = g_avg_distance_counter*(JOINT_NUM * 3 + 1);
	float avg_err = get_estimate_err_dist(cur_xyz_estimated[0], cur_xyz_ground_truth[0], estimate_error_distance_arr + tmp_index1 + 1);
	
	estimate_error_distance_arr[tmp_index1] = float(cur_gesture);
	estimate_joints_arr[tmp_index2] = float(cur_gesture);

	g_avg_distance += avg_err;
	float max_err = 0.0;
	cout << ">>>>>>>>>>>> current estimated joints (x,y,z) and errors (mm)" << endl;
	for (int i = 0; i < JOINT_NUM; ++i)
	{
		cout << "Joint #" << i << ": (";
		for (int j = 0; j < 3; ++j)
		{
			estimate_joints_arr[tmp_index2 + 1 + i * 3 + j] = cur_xyz_estimated[0][i * 3 + j];
			cout << cur_xyz_estimated[0][i * 3 + j] << " ";
		}
		float err = estimate_error_distance_arr[tmp_index1 + i + 1];
		cout << ") error=" << err << endl;
		if (max_err < err)
		{
			max_err = err;
		}
	}

	cout << "max error = " << max_err << endl;
	cout << "average error = " << avg_err << endl;

	if (max_err<=estimation_threshold)
	{
		++g_good_estimations;
	}
	++g_avg_distance_counter;
}

void load_pca() // Load PCA learned from other subjects
{
	char strDir[255];
	int var_num = JOINT_NUM * 3;
	// 1. eigenvalues
	sprintf(strDir, "%s\\%s\\train_eigenvalues.mat", PCA_DIR.c_str(), subject_names[TEST_SUBJECT].c_str());
	MATFile *pMatFile = matOpen(strDir, "r");
	mxArray *pMxArray = matGetVariable(pMatFile, "eigenvalues");
	if (!pMxArray)
	{
		return;
	}
	cout << "Load eigenvalues.mat successfully!" << endl;
	cur_fuser.fuser_pca.eigenvalues = cv::Mat(var_num, 1, CV_32FC1, mxGetData(pMxArray));
	matClose(pMatFile);

	// 2. eigenvectors
	sprintf(strDir, "%s\\%s\\train_eigenvectors.mat", PCA_DIR.c_str(), subject_names[TEST_SUBJECT].c_str());
	pMatFile = matOpen(strDir, "r");
	pMxArray = matGetVariable(pMatFile, "eigenvectors");
	if (!pMxArray)
	{
		return;
	}
	cout << "Load eigenvectors.mat successfully!" << endl;
	cur_fuser.fuser_pca.eigenvectors = cv::Mat(var_num, var_num, CV_32FC1, mxGetData(pMxArray));
	matClose(pMatFile);

	// 3. mean
	sprintf(strDir, "%s\\%s\\train_mean.mat", PCA_DIR.c_str(), subject_names[TEST_SUBJECT].c_str());
	pMatFile = matOpen(strDir, "r");
	pMxArray = matGetVariable(pMatFile, "mean");
	if (!pMxArray)
	{
		return;
	}
	cout << "Load mean.mat successfully!" << endl;
	cur_fuser.fuser_pca.mean = cv::Mat(1, var_num, CV_32FC1, mxGetData(pMxArray));
	matClose(pMatFile);

	//cout << "cur_fuser.fuser_pca.mean:" << endl;
	//cout << cur_fuser.fuser_pca.mean << endl;
}

bool image_step_forward()
{
	++cur_image;
	if (cur_image >= gesture_image_num[cur_gesture])
	{
		cur_image = 0;
		++cur_gesture;
		if (cur_gesture >= GESTURE_NUM)
		{
			cur_gesture = 0;
			return false;
		}
		loadCurrentGesture();
	}

	++cur_estimate_index;
	if (cur_estimate_index >= total_estimate_num)
	{
		cur_estimate_index = 0;
	}

	return true;
}

int _tmain(int argc, _TCHAR* argv[])
{
	SAFE_MAT_DELETE(heatmap_xy)
	SAFE_MAT_DELETE(heatmap_yz)
	SAFE_MAT_DELETE(heatmap_zx)

	// 1. Load output_heatmaps
	char full_path[256];
	// 1.1 XY
	sprintf(full_path, "%s\\%s\\est_heatmaps_%s.mat", HEATMAP_DIR.c_str(), subject_names[TEST_SUBJECT].c_str(), OBB_names[0].c_str());
	MATFile *pMatFile = matOpen(full_path, "r");
	mxArray *pMxArray = matGetVariable(pMatFile, HEATMAP_VAR_NAME.c_str());
	if (pMxArray)
	{
		cout << "Load est_heatmaps_" << OBB_names[0] << ".mat successfully!" << endl;
	}
	heatmap_xy = (double*)mxGetData(pMxArray);	// 18 x 18 x 21 x #test_images
	matClose(pMatFile);
	for (int i = 0; i < 4; ++i)
	{
		heatmap_dims[i] = mxGetDimensions(pMxArray)[i];
	}
	total_estimate_num = heatmap_dims[3];
	// 1.2 YZ
	sprintf(full_path, "%s\\%s\\est_heatmaps_%s.mat", HEATMAP_DIR.c_str(), subject_names[TEST_SUBJECT].c_str(), OBB_names[1].c_str());
	pMatFile = matOpen(full_path, "r");
	pMxArray = matGetVariable(pMatFile, HEATMAP_VAR_NAME.c_str());
	if (pMxArray)
	{
		cout << "Load est_heatmaps_" << OBB_names[1] << ".mat successfully!" << endl;
	}
	heatmap_yz = (double*)mxGetData(pMxArray);	// 18 x 18 x 21 x #test_images
	matClose(pMatFile);
	// 1.3 ZX
	sprintf(full_path, "%s\\%s\\est_heatmaps_%s.mat", HEATMAP_DIR.c_str(), subject_names[TEST_SUBJECT].c_str(), OBB_names[2].c_str());
	pMatFile = matOpen(full_path, "r");
	pMxArray = matGetVariable(pMatFile, HEATMAP_VAR_NAME.c_str());
	if (pMxArray)
	{
		cout << "Load est_heatmaps_" << OBB_names[2] << ".mat successfully!" << endl;
	}
	heatmap_zx = (double*)mxGetData(pMxArray);	// #test_images x 21 x 2
	matClose(pMatFile);

	// 2. load PCA
	load_pca();

	// 3. error distance
	int sz_error_distance = total_estimate_num*(JOINT_NUM + 1);
	g_avg_distance = 0.0;
	g_avg_distance_counter = 0;
	SAFE_DELETE_ARR(estimate_error_distance_arr);
	estimate_error_distance_arr = new float[sz_error_distance];

	int sz_estimate_joints = total_estimate_num*(JOINT_NUM * 3 + 1);
	SAFE_DELETE_ARR(estimate_joints_arr);
	estimate_joints_arr = new float[sz_estimate_joints];

	// 4. Init
	cur_gesture = 0;
	cur_image = 0;
	cur_estimate_index = 0;

	cur_fuser.PCA_SZ = 35;

	loadCurrentSubject();
	loadCurrentGesture();
	loadCurrentImage();

	// 5. Run
	while (image_step_forward())
	{
		loadCurrentImage();
	}

	// 6. Show Results
	g_avg_distance /= g_avg_distance_counter;
	cout << ">>>>>>>>>>>>>>>>>>>> average estimate error distance is: " << g_avg_distance << "(mm)" << endl;

	cout << 100.0*float(g_good_estimations) / float(g_avg_distance_counter) << "% frames with all joints error within " << estimation_threshold << " mm." << endl;
	return 0;
}

