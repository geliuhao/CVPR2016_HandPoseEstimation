//
//  Created by Liuhao Ge on 08/08/2016.
//

#include "stdafx.h"

#include "BoundingBox.h"

BoundingBox::BoundingBox()
{
	m_x_length = 0.0;
	m_y_length = 0.0;
	m_z_length = 0.0;
}

BoundingBox::~BoundingBox()
{
}

bool BoundingBox::create_box_OBB(float* xyz_data, int data_num, float* GT_xyz_data, int GT_data_num)
{
	m_xyz_data.clear();
	m_gt_xyz_data.clear();

	// 1. calculate center point
	Float3 center_pt(0.0, 0.0, 0.0);
	for (int i = 0; i < data_num; ++i)
	{
		if (fabs(xyz_data[(i << 1) + i + 2]) < FLT_EPSILON)	// depth == 0.0 then skip
		{
			continue;
		}
		m_xyz_data.push_back(Float4(xyz_data[(i << 1) + i],
			xyz_data[(i << 1) + i + 1], xyz_data[(i << 1) + i + 2], 1.0f));

		for (int j = 0; j < 3; ++j)
		{
			center_pt[j] += xyz_data[(i << 1) + i + j];			// 3 * i + j
		}		
	}
	for (int i = 0; i < GT_data_num; ++i)
	{
		m_gt_xyz_data.push_back(Float4(GT_xyz_data[(i << 1) + i],
			GT_xyz_data[(i << 1) + i + 1], GT_xyz_data[(i << 1) + i + 2], 1.0f));
		//for (int j = 0; j < 3; ++j)
		//{
		//	center_pt[j] += GT_xyz_data[(i << 1) + i + j];		// 3 * i + j
		//}
	}
	int pt_num = m_xyz_data.size();// +m_gt_xyz_data.size();
	if (pt_num<=0)
	{
		return false;
	}
	for (int i = 0; i < 3; ++i)
	{
		center_pt[i] /= pt_num;
	}

	// 2. calculate covariance matrix
	Float3x3 cov;
	for (int j = 0; j < 3; ++j)
	{
		for (int k = j; k < 3; ++k)
		{
			cov(j,k) = 0.0;
			for (int i = 0; i < m_xyz_data.size(); ++i)
			{
				cov(j, k) += m_xyz_data[i][j] * m_xyz_data[i][k];		//c[j] * c[k];
			}
			//for (int i = 0; i < m_gt_xyz_data.size(); ++i)
			//{
			//	cov(j, k) += m_gt_xyz_data[i][j] * m_gt_xyz_data[i][k];	//c[j] * c[k];
			//}
			cov(j, k) /= pt_num;	
			cov(j, k) -= (center_pt[j] * center_pt[k]);
		}
	}
	for (int j = 0; j < 3; ++j)	// covariance matrix is real symmetric matrix
	{
		for (int k = 0; k < j; ++k)
		{
			cov(j, k) = cov(k, j);
		}
	}
	//cout << "cov:" << endl;
	//cov.print();

	// 3. calculate Eigen vectors to determine x_axis, y_axis, z_axis
	Float3 x_axis, y_axis, z_axis;
	Float3x3 eigenvector_matrix;	// Eigen vectors
	double eigenvalue_array[3];		// Eigen values

	jacobbi(cov, eigenvector_matrix, eigenvalue_array);
	x_axis[0] = eigenvector_matrix(0,0);
	x_axis[1] = eigenvector_matrix(1,0);
	x_axis[2] = eigenvector_matrix(2,0);
	y_axis[0] = eigenvector_matrix(0,1);
	y_axis[1] = eigenvector_matrix(1,1);
	y_axis[2] = eigenvector_matrix(2,1);
	z_axis[0] = eigenvector_matrix(0,2);
	z_axis[1] = eigenvector_matrix(1,2);
	z_axis[2] = eigenvector_matrix(2,2);
	if (isZero(x_axis.length()) || isZero(y_axis.length()) || isZero(z_axis.length()))
	{
		cout << "isZero(x_axis.length()) || isZero(y_axis.length()) || isZero(z_axis.length())" << endl;
		x_axis = Float3(1.0, 0.0, 0.0);
		y_axis = Float3(0.0, 1.0, 0.0);
		z_axis = Float3(0.0, 0.0, 1.0);
	}
	else
	{
		x_axis.normalize();
		y_axis.normalize();
		Float3::cross(z_axis, y_axis, x_axis);		// cross
		if (z_axis[2]<0)
		{
			Float3::scale(z_axis, -1.0);
			if (x_axis[1]<0)		// set x axis upward
			{
				Float3::cross(x_axis, z_axis, y_axis);
			}
			else
			{
				Float3::cross(y_axis, x_axis, z_axis);
			}
		}
	}
	// 4. calculate length, width, height
	vector<double> aVec;
	vector<double> bVec;
	vector<double> cVec;
	for (int i = 0; i < pt_num; ++i)
	{
		Float3 ov;
		Float3::sub(ov, Float3(m_xyz_data[i][0], m_xyz_data[i][1], m_xyz_data[i][2]), center_pt);
		aVec.push_back(Float3::dot(ov, x_axis));	// dot
		bVec.push_back(Float3::dot(ov, y_axis));
		cVec.push_back(Float3::dot(ov, z_axis));
	}

	double aMin, aMax, bMin, bMax, cMin, cMax;

	aMin = *std::min_element(aVec.begin(), aVec.end());
	aMax = *std::max_element(aVec.begin(), aVec.end());
	bMin = *std::min_element(bVec.begin(), bVec.end());
	bMax = *std::max_element(bVec.begin(), bVec.end());
	cMin = *std::min_element(cVec.begin(), cVec.end());
	cMax = *std::max_element(cVec.begin(), cVec.end());

	double scale = 1.02;	// scale coefficient
	m_x_length = (aMax - aMin)*scale;
	m_y_length = (bMax - bMin)*scale;
	m_z_length = (cMax - cMin)*scale;

	// 5. adjust center point
	Float3 scale_x_axis(x_axis);
	Float3::scale(scale_x_axis, (aMin + aMax) / 2.0);
	Float3::add(center_pt, center_pt, scale_x_axis);

	Float3 scale_y_axis(y_axis);
	Float3::scale(scale_y_axis, (bMin + bMax) / 2.0);
	Float3::add(center_pt, center_pt, scale_y_axis);

	Float3 scale_z_axis(z_axis);
	Float3::scale(scale_z_axis, (cMin + cMax) / 2.0);
	Float3::add(center_pt, center_pt, scale_z_axis);

	// 6. get m_relative_trans
	m_relative_trans = Float4x4(x_axis[0], x_axis[1], x_axis[2], 0.0,
								y_axis[0], y_axis[1], y_axis[2], 0.0,
								z_axis[0], z_axis[1], z_axis[2], 0.0,
								center_pt[0], center_pt[1], center_pt[2], 1.0);
	//cout << "m_relative_trans:" << endl;
	//m_relative_trans.print();

	// 7. transform 3d points from world cs to BB cs
	Float4x4 inv_relative_trans;
	Float4x4::inverse(inv_relative_trans, m_relative_trans);
	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		Float4 tmp(m_xyz_data[i]);
		Float4::mult(m_xyz_data[i], inv_relative_trans, tmp);
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		Float4 tmp(m_gt_xyz_data[i]);
		Float4::mult(m_gt_xyz_data[i], inv_relative_trans, tmp);
	}

	return true;
}

bool BoundingBox::project_direct(cv::Mat* proj_im, cv::Point2f proj_uv[][JOINT_NUM], int sz)
// cv::Mat proj_im[3], cv::Point2f proj_uv[3][JOINT_NUM]
// 0: x-y, 1: y-z, 2: z-x
{
	for (int i = 0; i < 3; ++i)
	{
		proj_im[i].release();
	}
	int pad = int(sz*0.1);
	// 0. x-y
	//cv::Rect xy_bounding_box;
	//float xy_k = 0.0;
	proj_im[0] = cv::Mat::ones(sz, sz, CV_32F);
	if (m_x_length >= m_y_length)
	{
		proj_bounding_box[0].width = sz - (pad << 1);
		proj_bounding_box[0].height = int_rounding((proj_bounding_box[0].width*m_y_length) / m_x_length);
		proj_k[0] = float(proj_bounding_box[0].width) / m_x_length;
	}
	else
	{
		proj_bounding_box[0].height = sz - (pad << 1);
		proj_bounding_box[0].width = int_rounding((proj_bounding_box[0].height*m_x_length) / m_y_length);
		proj_k[0] = float(proj_bounding_box[0].height) / m_y_length;
	}
	proj_bounding_box[0].x = ((sz - proj_bounding_box[0].width) >> 1);
	proj_bounding_box[0].y = ((sz - proj_bounding_box[0].height) >> 1);
	cv::Mat xy_roi = proj_im[0](proj_bounding_box[0]);

	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		int xy_u = int_rounding(proj_k[0]*(m_xyz_data[i][0] + m_x_length / 2.0));
		int xy_v = int_rounding(proj_k[0]*(-m_xyz_data[i][1] + m_y_length / 2.0));

		if (xy_u < 0)	xy_u = 0;
		else if (xy_u >= proj_bounding_box[0].width)		xy_u = proj_bounding_box[0].width - 1;

		if (xy_v < 0)	xy_v = 0;
		else if (xy_v >= proj_bounding_box[0].height)	xy_v = proj_bounding_box[0].height - 1;

		float norm_depth = (m_xyz_data[i][2] + m_z_length / 2.0) / m_z_length;	// normalize 0-1
		if (norm_depth<0)
		{
			norm_depth = 0.0f;
		}
		if (xy_roi.at<float>(xy_v, xy_u) > norm_depth)		// set the nearest point
		{
			xy_roi.at<float>(xy_v, xy_u) = norm_depth;
		}
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		proj_uv[0][i].x = int_rounding(proj_k[0]*(m_gt_xyz_data[i][0] + m_x_length / 2.0)) + proj_bounding_box[0].x;
		proj_uv[0][i].y = int_rounding(proj_k[0]*(-m_gt_xyz_data[i][1] + m_y_length / 2.0)) + proj_bounding_box[0].y;

		if (proj_uv[0][i].x < 0)	proj_uv[0][i].x = 0;
		else if (proj_uv[0][i].x >= sz)	proj_uv[0][i].x = sz - 1;

		if (proj_uv[0][i].y < 0)	proj_uv[0][i].y = 0;
		else if (proj_uv[0][i].y >= sz)	proj_uv[0][i].y = sz - 1;
	}

	// 1. y-z
	//cv::Rect yz_bounding_box;
	//float yz_k = 0.0;
	proj_im[1] = cv::Mat::ones(sz, sz, CV_32F);
	if (m_y_length >= m_z_length)
	{
		proj_bounding_box[1].width = sz - (pad << 1);
		proj_bounding_box[1].height = int_rounding((proj_bounding_box[1].width*m_z_length) / m_y_length);
		proj_k[1] = float(proj_bounding_box[1].width) / m_y_length;
	}
	else
	{
		proj_bounding_box[1].height = sz - (pad << 1);
		proj_bounding_box[1].width = int_rounding((proj_bounding_box[1].height*m_y_length) / m_z_length);
		proj_k[1] = float(proj_bounding_box[1].height) / m_z_length;
	}
	proj_bounding_box[1].x = ((sz - proj_bounding_box[1].width) >> 1);
	proj_bounding_box[1].y = ((sz - proj_bounding_box[1].height) >> 1);
	cv::Mat yz_roi = proj_im[1](proj_bounding_box[1]);

	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		int yz_u = int_rounding(proj_k[1]*(-m_xyz_data[i][1] + m_y_length / 2.0));
		int yz_v = int_rounding(proj_k[1]*(m_xyz_data[i][2] + m_z_length / 2.0));
		
		if (yz_u < 0)	yz_u = 0;
		else if (yz_u >= proj_bounding_box[1].width)		yz_u = proj_bounding_box[1].width - 1;

		if (yz_v < 0)	yz_v = 0;
		else if (yz_v >= proj_bounding_box[1].height)	yz_v = proj_bounding_box[1].height - 1;

		float norm_depth = (-m_xyz_data[i][0] + m_x_length / 2.0) / m_x_length;//(m_xyz_data[i][0] + m_x_length / 2.0) / m_x_length;	// normalize 0-1
		if (norm_depth<0)
		{
			norm_depth = 0.0f;
		}
		if (yz_roi.at<float>(yz_v, yz_u) > norm_depth)		// set the nearest point
		{
			yz_roi.at<float>(yz_v, yz_u) = norm_depth;
		}
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		proj_uv[1][i].x = int_rounding(proj_k[1]*(-m_gt_xyz_data[i][1] + m_y_length / 2.0)) + proj_bounding_box[1].x;
		proj_uv[1][i].y = int_rounding(proj_k[1]*(m_gt_xyz_data[i][2] + m_z_length / 2.0)) + proj_bounding_box[1].y;

		if (proj_uv[1][i].x < 0)	proj_uv[1][i].x = 0;
		else if (proj_uv[1][i].x >= sz)	proj_uv[1][i].x = sz - 1;

		if (proj_uv[1][i].y < 0)	proj_uv[1][i].y = 0;
		else if (proj_uv[1][i].y >= sz)	proj_uv[1][i].y = sz - 1;
	}

	// 2. z-x
	//cv::Rect zx_bounding_box;
	//float zx_k = 0.0;
	proj_im[2] = cv::Mat::ones(sz, sz, CV_32F);
	if (m_z_length >= m_x_length)
	{
		proj_bounding_box[2].width = sz - (pad << 1);
		proj_bounding_box[2].height = int_rounding((proj_bounding_box[2].width*m_x_length) / m_z_length);
		proj_k[2] = float(proj_bounding_box[2].width) / m_z_length;
	}
	else
	{
		proj_bounding_box[2].height = sz - (pad << 1);
		proj_bounding_box[2].width = int_rounding((proj_bounding_box[2].height*m_z_length) / m_x_length);
		proj_k[2] = float(proj_bounding_box[2].height) / m_x_length;
	}
	proj_bounding_box[2].x = ((sz - proj_bounding_box[2].width) >> 1);
	proj_bounding_box[2].y = ((sz - proj_bounding_box[2].height) >> 1);
	cv::Mat zx_roi = proj_im[2](proj_bounding_box[2]);

	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		int zx_u = int_rounding(proj_k[2]*(m_xyz_data[i][2] + m_z_length / 2.0));
		int zx_v = int_rounding(proj_k[2]*(m_xyz_data[i][0] + m_x_length / 2.0));

		if (zx_u < 0)	zx_u = 0;
		else if (zx_u >= proj_bounding_box[2].width)		zx_u = proj_bounding_box[2].width - 1;

		if (zx_v < 0)	zx_v = 0;
		else if (zx_v >= proj_bounding_box[2].height)	zx_v = proj_bounding_box[2].height - 1;

		float norm_depth = (-m_xyz_data[i][1] + m_y_length / 2.0) / m_y_length;	// normalize 0-1
		if (norm_depth<0)
		{
			norm_depth = 0.0f;
		}
		if (zx_roi.at<float>(zx_v, zx_u) > norm_depth)		// set the nearest point
		{
			zx_roi.at<float>(zx_v, zx_u) = norm_depth;
		}
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		proj_uv[2][i].x = int_rounding(proj_k[2]*(m_gt_xyz_data[i][2] + m_z_length / 2.0)) + proj_bounding_box[2].x;
		proj_uv[2][i].y = int_rounding(proj_k[2]*(m_gt_xyz_data[i][0] + m_x_length / 2.0)) + proj_bounding_box[2].y;

		if (proj_uv[2][i].x < 0)	proj_uv[2][i].x = 0;
		else if (proj_uv[2][i].x >= sz)	proj_uv[2][i].x = sz - 1;

		if (proj_uv[2][i].y < 0)	proj_uv[2][i].y = 0;
		else if (proj_uv[2][i].y >= sz)	proj_uv[2][i].y = sz - 1;
	}

	clock_t t0 = clock();

	cv::medianBlur(proj_im[0], proj_im[0], 5);
	cv::medianBlur(proj_im[1], proj_im[1], 5);
	cv::medianBlur(proj_im[2], proj_im[2], 5);

	return true;
}

void BoundingBox::jacobbi(const Float3x3& input_mat, Float3x3& v, double* pArray)
{
	int p, q, j, ind, n;
	double dsqr, d1, d2, thr, dv1, dv2, dv3, dmu, dga, st, ct;
	double eps = 0.00000001;
	int* iZ; //add 2002.8.27

	Float3x3 CA(input_mat);
	n = 3;

	//add 2002.8.27
	iZ = new int[n];

	for (p = 0; p < n; p++)
	for (q = 0; q < n; q++)
		v(p,q) = (p == q) ? 1.0 : 0;

	dsqr = 0;
	for (p = 1; p < n; p++)
	for (q = 0; q < p; q++)
		dsqr += 2 * CA(p, q) * CA(p, q);
	d1 = sqrt(dsqr);
	d2 = eps / n * d1;
	thr = d1;
	ind = 0;
	do {
		thr = thr / n;
		while (!ind) {
			for (q = 1; q < n; q++)
			for (p = 0; p < q; p++)
			if (fabs(CA(p, q)) >= thr) {
				ind = 1;
				dv1 = CA(p, p);
				dv2 = CA(p, q);
				dv3 = CA(q, q);
				dmu = 0.5 * (dv1 - dv3);
				double dls = sqrt(dv2 * dv2 + dmu * dmu);
				if (fabs(dmu) < 0.00000000001) dga = -1;
				//if ( dmu == 0.0 ) dga = -1.0 ;
				else dga = (dmu < 0) ? (dv2 / dls) : (-dv2 / dls);
				st = dga / sqrt(2 * (1 + sqrt(1 - dga * dga)));
				ct = sqrt(1 - st * st);
				for (int l = 0; l < n; l++) {
					dsqr = CA(l,p) * ct - CA(l,q) * st;
					CA(l, q) = CA(l, p) * st + CA(l, q) * ct;
					CA(l, p) = dsqr;
					dsqr = v(l, p) * ct - v(l, q) * st;
					v(l, q) = v(l, p) * st + v(l, q) * ct;
					v(l, p) = dsqr;
				}
				for (int l = 0; l < n; l++) {
					CA(p, l) = CA(l, p);
					CA(q, l) = CA(l, q);
				}
				CA(p,p) = dv1 * ct * ct + dv3 * st * st - 2 * dv2 * st * ct;
				CA(q,q) = dv1 * st * st + dv3 * ct * ct + 2 * dv2 * st * ct;
				CA(p,q) = CA(q,p) = 0.0;
			}
			if (ind) ind = 0;
			else break;
		}
	} while (thr > d2);
	for (int l = 0; l < n; l++) {
		pArray[l] = CA(l,l);
		iZ[l] = l;
	}
	double dTemp;
	int i, k;

	for (i = 0; i < n; i++){
		//dmax = pArray[i];
		for (j = i + 1; j < n; j++){
			if (pArray[i] < pArray[j]){
				dTemp = pArray[i];
				pArray[i] = pArray[j];
				pArray[j] = dTemp;
				k = iZ[i];
				iZ[i] = iZ[j];
				iZ[j] = k;
			}
		}
	}
	CA = v;

	for (j = 0; j < n; j++)
		for (i = 0; i < n; i++)
			v(i,j) = CA(i,iZ[j]);

	delete[] iZ;
}

double BoundingBox::get_z_value(double x_val, double y_val)
{

	return 0.0;
}

void BoundingBox::get_project_points(float* xyz_data, int data_num, float* xy_data, float* yz_data, float* zx_data)
{
	Float4x4 inv_relative_trans;
	Float4x4::inverse(inv_relative_trans, m_relative_trans);

	for (int i = 0; i < data_num; ++i)
	{
		Float4 cur_xyz_data(xyz_data[(i << 1) + i], xyz_data[(i << 1) + i + 1], xyz_data[(i << 1) + i + 2], 1.0f);
		if (fabs(xyz_data[(i << 1) + i + 2]) < FLT_EPSILON)	// depth == 0.0 then skip
		{
			continue;
		}
		Float4 tmp(cur_xyz_data);
		Float4::mult(cur_xyz_data, inv_relative_trans, tmp);

		Float4 xy_proj(cur_xyz_data);
		xy_proj[2] = -m_z_length / 2.0;
		Float4 tmp_xy(xy_proj);
		Float4::mult(xy_proj, m_relative_trans, tmp_xy);

		Float4 yz_proj(cur_xyz_data);
		yz_proj[0] = m_x_length / 2.0;
		Float4 tmp_yz(yz_proj);
		Float4::mult(yz_proj, m_relative_trans, tmp_yz);

		Float4 zx_proj(cur_xyz_data);
		zx_proj[1] = m_y_length / 2.0;
		Float4 tmp_zx(zx_proj);
		Float4::mult(zx_proj, m_relative_trans, tmp_zx);

		for (int j = 0; j < 3; ++j)
		{
			xy_data[i * 3 + j] = xy_proj[j];
			yz_data[i * 3 + j] = yz_proj[j];
			zx_data[i * 3 + j] = zx_proj[j];
		}
	}
}

jtil::math::Float3 BoundingBox::get_yaw_pitch_roll()
{
	Float3 angles(0.0, 0.0, 0.0);	// yaw-alpha, pitch-beta, roll-garma

	angles[0] = atan2(-m_relative_trans(2, 0), sqrt(pow(m_relative_trans(0, 0), 2) + pow(m_relative_trans(1, 0), 2)));
	angles[1] = atan2(m_relative_trans(1, 0), m_relative_trans(0, 0));
	angles[2] = atan2(m_relative_trans(2, 1), m_relative_trans(2, 2));

	return angles;
}
