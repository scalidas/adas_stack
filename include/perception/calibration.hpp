#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace adas::calibration {
// Display a window with 4 source points that can be moved by the user using
// the mouse. User can write down the 4 corner points
std::vector<cv::Point2f> interactive_calibration(std::string image_path);

} // namespace adas::calibration
