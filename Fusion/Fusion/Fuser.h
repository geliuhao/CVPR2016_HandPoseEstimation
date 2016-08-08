//
//  Created by Liuhao Ge on 08/08/2016.
//

#pragma once

#include "BoundingBox.h"
#include <cv.h>
#include <highgui.h>

class Fuser
{
public:
	Fuser();
	~Fuser();

	cv::PCA fuser_pca;
	int PCA_SZ;

	double bounding_box_x[3];
	double bounding_box_y[3];
	double proj_k[3];
	double bounding_box_width[3];
	double bounding_box_height[3];

	BoundingBox bounding_box_3D;

	vector<cv::Mat> heatmaps_vec;

	int estimate_gauss_failed_cnt;

	void fuse(float* estimate_xyz);			// return xyz in world cs (96 x 96 3d space) - gauss covariance + PCA
	void fuse_sub(float* estimate_xyz);		// return xyz in world cs (96 x 96 3d space) - mean-shift

private:
	double bounding_box_x_18[3];
	double bounding_box_y_18[3];
	double bounding_box_width_18[3];
	double bounding_box_height_18[3];

	vector<vector<Float4> > pca_eigen_vecs_bb;	// PCA_SZ x HEAT_NUM
	vector<Float4> pca_means_bb;		// HEAT_NUM

	vector<Float4> joints_means_bb;		// HEAT_NUM
	vector<Float4> joints_variance_bb;	// HEAT_NUM
	vector<Float3x3> joints_covariance_bb;	// HEAT_NUM
	vector<Float3x3> joints_inv_covariance_bb;	// HEAT_NUM

	Float4 estimate_joint_xyz(int joint_i);	// return xyz in BB cs (96 x 96 3d space)


	void _2d_3d(int view_type, int u, int v, cv::Point3d& pt);	// view_type: 0-xy, 1-yz, 2-zx; 18 x 18 3d space

	void xy_3d(int u, int v, double& x, double& y);	// 18 x 18 u, v ---> xyz in 18 x 18 3d space
	void yz_3d(int u, int v, double& y, double& z); // 18 x 18 u, v ---> xyz in 18 x 18 3d space
	void zx_3d(int u, int v, double& z, double& x); // 18 x 18 u, v ---> xyz in 18 x 18 3d space
	void xyz_18_96(const Float4& xyz_18, Float4& xyz_96);	// xyz in 18 x 18 3d space ---> xyz in 96 x 96 3d space
	void xyz_96_18(const Float4& xyz_96, Float4& xyz_18);	// xyz in 96 x 96 3d space ---> xyz in 18 x 18 3d space

	void _3d_xy(double x, double y, int& u, int& v); // xyz in 18 x 18 3d space ---> 18 x 18 u, v
	void _3d_yz(double y, double z, int& u, int& v); // xyz in 18 x 18 3d space ---> 18 x 18 u, v
	void _3d_zx(double z, double x, int& u, int& v); // xyz in 18 x 18 3d space ---> 18 x 18 u, v

	void convert_PCA_wld_to_BB();	// convert PCA in world cs to BB cs

	// estimate the mean and variance (in 18 x 18 3d space) of the gaussian distribution for each joint point
	bool estimate_gauss_mean_covariance(int joint_i, Float4& mean_18, Float3x3& covariance_18);	// get covariance matrix

	float fuse_confidence(float conf_xy, float conf_yz, float conf_zx);
};

