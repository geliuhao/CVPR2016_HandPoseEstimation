//
//  Created by Liuhao Ge on 08/08/2016.
//

#pragma once

#include <vector>
#include <cv.h>
#include "math/math_types.h"

using namespace std;
using namespace jtil::math;

#define BOX_VERTEX_NUM 8
#define BOX_LINE_NUM 12

#define JOINT_NUM 21

class BoundingBox
{
public:
	BoundingBox();
	~BoundingBox();

	bool create_box_OBB(float* xyz_data, int data_num, float* GT_xyz_data, int GT_data_num);

	// cv::Mat proj_im[3], cv::Point2f proj_uv[3][JOINT_NUM]
	bool project_direct(cv::Mat* proj_im, cv::Point2f proj_uv[][JOINT_NUM], int sz);

	double get_x_length() const				// get_x_length
	{
		return m_x_length;
	}
	void set_x_length(double x_length)	// set_x_length
	{
		m_x_length = x_length;
	}

	double get_y_length() const				// get_y_length
	{
		return m_y_length;
	}
	void set_y_length(double y_length)	// set_y_length
	{
		m_y_length = y_length;
	}

	double get_z_length() const				// get_z_length
	{
		return m_z_length;
	}
	void set_z_length(double z_length)	// set_z_length
	{
		m_z_length = z_length;
	}

	Float4x4& get_relative_trans()
	{
		return m_relative_trans;
	}

	void set_relative_trans(const Float4x4& relative_trans)
	{
		m_relative_trans = relative_trans;
	}

	cv::Rect* get_proj_bounding_box()
	{
		return proj_bounding_box;
	}
	float* get_proj_k()
	{
		return proj_k;
	}

	double get_z_value(double x_val, double y_val);	// all are in BB cs

	void get_project_points(float* xyz_data, int data_num, float* xy_data, float* yz_data, float* zx_data);

	Float3 get_yaw_pitch_roll();
	
	vector<Float4> get_gt_xyz_data()
	{
		return m_gt_xyz_data;
	}

private:
	double m_x_length;
	double m_y_length;
	double m_z_length;
	Float4x4 m_relative_trans;	// world cs to BB cs

	vector<Float4> m_xyz_data;		// point cloud (in BB cs)
	vector<Float4> m_gt_xyz_data;	// ground truth points (in BB cs)

	cv::Rect proj_bounding_box[3];
	float proj_k[3];

	int int_rounding(float val)
	{
		return int(val + 0.5);
	}

	bool isZero(float val)
	{
		if (fabs(val) <= FLT_EPSILON)
		{
			return true;
		}
		return false;
	}

	void jacobbi(const Float3x3& input_mat, Float3x3& v, double* pArray);
};

