#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace adas::calibration {
// Display a window with 4 source points that can be moved by the user using
// the mouse. Saves extrinsic calibration homography matrix to output_path.
int run_extrinsic_calibration(
    const std::string &image_path,
    const std::string &output_path = "config/calibration/extrinsic_calibration.xml");

// Save extrinsic calibration (src_pts, dst_pts, homography matrix H) to XML file
bool save_extrinsic_calibration(const std::string &filename,
                                const std::vector<cv::Point2f> &src_pts,
                                const std::vector<cv::Point2f> &dst_pts,
                                const cv::Mat &H);

// Load extrinsic calibration (homography matrix H, optional src_pts & dst_pts) from XML file
bool load_extrinsic_calibration(const std::string &filename, cv::Mat &H,
                                std::vector<cv::Point2f> *src_pts = nullptr,
                                std::vector<cv::Point2f> *dst_pts = nullptr);

// Compute the 3x3 transformation matrix from camera coordinate frame [x_depth, y_lat, z_vert]
// to ground plane reference frame [X_fwd, Y_right, Z_up] using extrinsic calibration and intrinsics
bool compute_camera_to_ground_transform(
    const std::string &extrinsics_path,
    const cv::Mat &camera_matrix,
    const cv::Mat &dist_coeffs,
    cv::Mat &ground_transform,
    int calib_w = 640,
    int calib_h = 480);

} // namespace adas::calibration
