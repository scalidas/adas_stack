#pragma once

#include "config/config.hpp"
#include "core/core_types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace adas::perception {

// Return a vector of all the lanes present in the BEV image using configuration parameters
std::vector<adas::core::PolynomialLaneBoundary>
detect_lanes(cv::Mat &bev,
             const adas::config::PerceptionConfig &config = adas::config::PerceptionConfig{});

} // namespace adas::perception