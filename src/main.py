import numpy as np
from numpy.linalg import norm, inv, pinv
from utils import load_data
from slam_utils import motion_model_prediction, landmark_mapping, visual_slam
from visualize_utils import compare_result


if __name__ == '__main__':
	# read data
	# you can downsample the features size to achieve time reduction
	filename = "./data/0034.npz"  # 22, 27, 34
	t, features, linear_velocity, angular_velocity, K, b, cam_T_imu = load_data(filename)

	# IMU Localization via EKF Prediction
	i_T_w, w_T_i = motion_model_prediction(t, linear_velocity, angular_velocity, w_scale=10e-7)

	# Landmark Mapping via EKF Update
	landmarks = landmark_mapping(features, i_T_w, K, b, cam_T_imu)

	# Visual-Inertial SLAM
	slam_iTw, slam_wTi, slam_landmarks = visual_slam(t, linear_velocity, angular_velocity, features, K, b, cam_T_imu)

	# visualization
	compare_result(w_T_i, slam_wTi, landmarks, slam_landmarks)
