//
//  Created by Liuhao Ge on 01/10/2017.
//

// Input: xyz_pts: pointer to an array containin 21*3 values which are x, y, z values for 21 hand joints, the order is consistent with that in the dataset
// Outputs: angles: yaw, pitch, roll angles

#include "math/math_types.h"

Float3 calculate_yaw_pitch_roll(const float* xyz_pts)
{
	Float3 y_axis, tmp_vec1, tmp_vec2, tmp_vec3, tmp_z1, tmp_z2, tmp_z3, z_axis, x_axis;
	for (int i = 0; i < 3; ++i)
	{
		y_axis[i] = xyz_pts[5*3+i] - xyz_pts[i];
		tmp_vec1[i] = xyz_pts[1*3+i] - xyz_pts[i];
		tmp_vec2[i] = xyz_pts[9*3+i] - xyz_pts[i];
		tmp_vec3[i] = xyz_pts[13*3+i] - xyz_pts[i];
	}

	if (y_axis.length() <= 1e-9)
	{
		return Float3(0.0, 0.0, 0.0);
	}

	y_axis.normalize();

	int cnt = 0;
	z_axis = Float3(0.0, 0.0, 0.0);
	if (tmp_vec1.length() > 1e-9)
	{
		tmp_vec1.normalize();
		Float3::cross(tmp_z1, tmp_vec1, y_axis);
		if (tmp_z1.length() > 1e-9)
		{
			++cnt;
			tmp_z1.normalize();
			Float3::add(z_axis, z_axis, tmp_z1);
		}
	}
	if (tmp_vec2.length() > 1e-9)
	{
		tmp_vec2.normalize();
		Float3::cross(tmp_z2, y_axis, tmp_vec2);
		if (tmp_z2.length() > 1e-9)
		{
			++cnt;
			tmp_z2.normalize();
			Float3::add(z_axis, z_axis, tmp_z2);
		}
	}
	if (tmp_vec3.length() > 1e-9)
	{
		tmp_vec3.normalize();
		Float3::cross(tmp_z3, y_axis, tmp_vec3);
		if (tmp_z3.length() > 1e-9)
		{
			++cnt;
			tmp_z3.normalize();
			Float3::add(z_axis, z_axis, tmp_z3);
		}
	}

	if (cnt==0 || z_axis.length() <= 1e-9)
	{
		return Float3(0.0, 0.0, 0.0);
	}

	Float3::scale(z_axis, 1.0 / float(cnt));
	z_axis.normalize();
	Float3::cross(x_axis, y_axis, z_axis);
	x_axis.normalize();

	Float3 angles(0.0, 0.0, 0.0);	// yaw, pitch, roll

	angles[0] = atan2(-x_axis[2], sqrt(pow(x_axis[0], 2) + pow(x_axis[1], 2)));
	angles[1] = atan2(x_axis[1], x_axis[0]);
	angles[2] = atan2(y_axis[2], z_axis[2]);

	return angles;
}