#include "perception/lane_detection.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <random>
#include <vector>

namespace adas::perception {

namespace {

std::optional<adas::core::PolynomialLaneBoundary>
fitPolynomialSVD(const std::vector<cv::Point2f> &points) {
  if (points.size() < 3)
    return std::nullopt;

  const int n = static_cast<int>(points.size());
  Eigen::MatrixXd A(n, 3);
  Eigen::VectorXd b(n);

  for (int i = 0; i < n; ++i) {
    // Map: x = forward distance (vertical y in BEV), y = lateral offset (horizontal x in BEV)
    double x = points[i].y;
    double y = points[i].x;
    A(i, 0) = 1.0;
    A(i, 1) = x;
    A(i, 2) = x * x;
    b(i) = y;
  }

  Eigen::Vector3d c =
      A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  return adas::core::PolynomialLaneBoundary{c(0), c(1), c(2)};
}

/*
RANSAC based polynomial fitting for lane lines
*/
std::vector<adas::core::PolynomialLaneBoundary> fitLaneRANSAC(
    std::vector<cv::Point2f> input_points,
    int max_iters = 150,
    float inlier_threshold_px = 10.0f,
    int min_inlier_count = 20) {
  thread_local std::mt19937 rng(std::random_device{}());

  std::vector<adas::core::PolynomialLaneBoundary> lanes;
  while (input_points.size() >= static_cast<size_t>(min_inlier_count)) {
    std::uniform_int_distribution<size_t> dist(0, input_points.size() - 1);

    std::vector<cv::Point2f> best_inliers;
    std::vector<cv::Point2f> best_outliers;
    size_t max_inlier_count = 0;

    for (int iter = 0; iter < max_iters; iter++) {
      std::vector<cv::Point2f> sample;
      for (int k = 0; k < 3; ++k) {
        sample.push_back(input_points[dist(rng)]);
      }

      auto candidate = fitPolynomialSVD(sample);
      if (!candidate.has_value())
        continue;

      std::vector<cv::Point2f> current_inliers;
      std::vector<cv::Point2f> current_outliers;

      current_inliers.reserve(input_points.size());
      current_outliers.reserve(input_points.size());

      for (const auto &pt : input_points) {
        double x = pt.y;
        double y_actual = pt.x;
        double y_pred = candidate->evaluate(x);

        if (std::abs(y_actual - y_pred) < inlier_threshold_px) {
          current_inliers.push_back(pt);
        } else {
          current_outliers.push_back(pt);
        }
      }

      if (current_inliers.size() > max_inlier_count) {
        max_inlier_count = current_inliers.size();
        best_inliers = std::move(current_inliers);
        best_outliers = std::move(current_outliers);
      }
    }

    if (best_inliers.size() < static_cast<size_t>(min_inlier_count)) {
      break;
    }

    auto lane = fitPolynomialSVD(best_inliers);
    if (!lane.has_value()) {
      break;
    }
    lanes.push_back(lane.value());

    input_points = std::move(best_outliers);
  }

  return lanes;
}

} // namespace

std::vector<adas::core::PolynomialLaneBoundary>
detect_lanes(cv::Mat &bev, const adas::config::PerceptionConfig &config) {
  if (bev.empty())
    return {};

  // Convert to grayscale & threshold using config.white_pixel_threshold
  cv::Mat gray, binary;
  if (bev.channels() == 3) {
    cv::cvtColor(bev, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = bev;
  }
  cv::threshold(gray, binary, config.white_pixel_threshold, 255, cv::THRESH_BINARY);

  // Extract all candidate non-zero edge pixels from BEV
  std::vector<cv::Point> non_zero_locs;
  cv::findNonZero(binary, non_zero_locs);

  std::vector<cv::Point2f> input_points(non_zero_locs.begin(), non_zero_locs.end());

  std::vector<adas::core::PolynomialLaneBoundary> detected_boundaries;

  // Fit boundaries using RANSAC parameters from config
  auto lanes = fitLaneRANSAC(input_points, config.ransac_iterations,
                             config.ransac_inlier_threshold, config.min_inlier_count);
  if (!lanes.empty()) {
    detected_boundaries.insert(detected_boundaries.end(), lanes.begin(), lanes.end());
  }

  return detected_boundaries;
}

} // namespace adas::perception
