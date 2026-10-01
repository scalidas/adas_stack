#include "perception/lane_detection.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <random>
#include <sstream>
#include <string>
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
    // Map: x = forward distance (vertical y in BEV), y = lateral offset
    // (horizontal x in BEV)
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
std::vector<adas::core::PolynomialLaneBoundary>
fitLaneRANSAC(std::vector<cv::Point2f> input_points, int max_iters = 150,
              float inlier_threshold_px = 10.0f, int min_inlier_count = 20) {
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

// Helper to check if image_path starts with "webcam" and parse index n
bool parse_webcam_index(const std::string &path, int &camera_id) {
  std::string lower_path = path;
  std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_path.rfind("webcam", 0) == 0) {
    std::string remaining = lower_path.substr(6);
    std::stringstream ss(remaining);
    if (ss >> camera_id) {
      return true;
    }
    camera_id = 0; // Default to webcam 0 if no number specified
    return true;
  }
  return false;
}

// Helper to load camera intrinsics and perform undistortion
struct CameraIntrinsics {
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  int calib_width{0};
  int calib_height{0};
  bool use_fisheye{false};
  bool loaded{false};

  bool load(const std::string &path) {
    if (path.empty())
      return false;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "[Intrinsics] Failed to open intrinsics file: " << path
                << std::endl;
      return false;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs["image_width"] >> calib_width;
    fs["image_height"] >> calib_height;
    int fisheye_val = 0;
    fs["fisheye_model"] >> fisheye_val;
    use_fisheye = (fisheye_val != 0);

    if (camera_matrix.empty() || dist_coeffs.empty()) {
      std::cerr << "[Intrinsics] Invalid intrinsics file content: " << path
                << std::endl;
      return false;
    }
    loaded = true;
    std::cout << "[Intrinsics] Loaded intrinsics from: " << path << std::endl;
    return true;
  }

  cv::Mat get_scaled_K(const cv::Size &current_size) const {
    if (!loaded || calib_width <= 0 || calib_height <= 0)
      return camera_matrix;
    if (current_size.width == calib_width &&
        current_size.height == calib_height) {
      return camera_matrix;
    }
    cv::Mat K = camera_matrix.clone();
    double sx = static_cast<double>(current_size.width) / calib_width;
    double sy = static_cast<double>(current_size.height) / calib_height;
    K.at<double>(0, 0) *= sx;
    K.at<double>(0, 2) *= sx;
    K.at<double>(1, 1) *= sy;
    K.at<double>(1, 2) *= sy;
    return K;
  }

  void undistort(const cv::Mat &src, cv::Mat &dst) const {
    if (!loaded) {
      dst = src.clone();
      return;
    }
    cv::Mat K = get_scaled_K(src.size());
    if (use_fisheye) {
      cv::Mat new_K;
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
          K, dist_coeffs, src.size(), cv::Matx33d::eye(), new_K, 1);
      cv::fisheye::undistortImage(src, dst, K, dist_coeffs, new_K);
    } else {
      cv::undistort(src, dst, K, dist_coeffs);
    }
  }

  void init_rectify_maps(const cv::Size &size, cv::Mat &map1,
                         cv::Mat &map2) const {
    if (!loaded)
      return;
    cv::Mat K = get_scaled_K(size);
    if (use_fisheye) {
      cv::Mat new_K;
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
          K, dist_coeffs, size, cv::Matx33d::eye(), new_K, 1);
      cv::fisheye::initUndistortRectifyMap(K, dist_coeffs, cv::Matx33d::eye(),
                                           new_K, size, CV_16SC2, map1, map2);
    } else {
      cv::initUndistortRectifyMap(K, dist_coeffs, cv::Mat(), K, size, CV_16SC2,
                                  map1, map2);
    }
  }
};

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
  cv::threshold(gray, binary, config.white_pixel_threshold, 255,
                cv::THRESH_BINARY);

  // Extract all candidate non-zero edge pixels from BEV
  std::vector<cv::Point> non_zero_locs;
  cv::findNonZero(binary, non_zero_locs);

  std::vector<cv::Point2f> input_points(non_zero_locs.begin(),
                                        non_zero_locs.end());

  std::vector<adas::core::PolynomialLaneBoundary> detected_boundaries;

  // Fit boundaries using RANSAC parameters from config
  auto lanes =
      fitLaneRANSAC(input_points, config.ransac_iterations,
                    config.ransac_inlier_threshold, config.min_inlier_count);
  if (!lanes.empty()) {
    detected_boundaries.insert(detected_boundaries.end(), lanes.begin(),
                               lanes.end());
  }

  return detected_boundaries;
}

int detect_lanes_demo(const adas::config::PerceptionConfig &config,
                      cv::Mat ipm) {
  int camera_id = 0;
  bool is_webcam = parse_webcam_index(config.image_path, camera_id);

  CameraIntrinsics intrinsics;
  bool has_intrinsics = intrinsics.load(config.intrinsics_path);

  if (is_webcam) {
    std::cout << "[Live Demo] Opening Webcam ID: " << camera_id << std::endl;
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open webcam ID: " << camera_id << std::endl;
      return -1;
    }

    std::cout << "Running live webcam lane detection. Press ESC or 'q' to quit."
              << std::endl;

    cv::Mat frame, bev, gray, binary, debug_vis;
    cv::Mat map1, map2;
    bool maps_initialized = false;

    while (true) {
      cap >> frame;
      if (frame.empty()) {
        std::cerr << "Empty frame received from webcam." << std::endl;
        break;
      }

      if (has_intrinsics) {
        if (!maps_initialized) {
          intrinsics.init_rectify_maps(frame.size(), map1, map2);
          maps_initialized = true;
        }
        cv::Mat frame_undistorted;
        cv::remap(frame, frame_undistorted, map1, map2, cv::INTER_LINEAR);
        frame = frame_undistorted;
      }

      cv::resize(frame, frame, cv::Size(640, 480));

      cv::warpPerspective(frame, bev, ipm, cv::Size(640, 480));

      // Run lane detection
      std::vector<adas::core::PolynomialLaneBoundary> boundaries =
          adas::perception::detect_lanes(bev, config);

      // Compute binary threshold visualization
      if (bev.channels() == 3) {
        cv::cvtColor(bev, gray, cv::COLOR_BGR2GRAY);
      } else {
        gray = bev;
      }
      cv::threshold(gray, binary, config.white_pixel_threshold, 255,
                    cv::THRESH_BINARY);

      debug_vis = bev.clone();

      for (size_t i = 0; i < boundaries.size(); ++i) {
        const auto &lane = boundaries[i];
        for (int y_coord = 0; y_coord < 480; y_coord += 2) {
          double x_coord = lane.evaluate(y_coord);
          if (x_coord >= 0 && x_coord < 640) {
            cv::circle(debug_vis, cv::Point(static_cast<int>(x_coord), y_coord),
                       2, cv::Scalar(0, 255, 0), -1);
          }
        }
      }

      cv::imshow("Original Frame", frame);
      cv::imshow("BEV Binary", binary);
      cv::imshow("Fitted Lane (BEV)", debug_vis);

      int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        break;
      }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
  }

  // Static image demo mode
  cv::Mat frame = cv::imread(config.image_path);
  if (frame.empty()) {
    std::cerr << "Failed to load image: " << config.image_path << std::endl;
    return -1;
  }

  if (has_intrinsics) {
    cv::Mat frame_undistorted;
    intrinsics.undistort(frame, frame_undistorted);
    frame = frame_undistorted;
  }

  cv::resize(frame, frame, cv::Size(640, 480));

  cv::Mat bev;
  cv::warpPerspective(frame, bev, ipm, cv::Size(640, 480));

  // Run lane detection using loaded configuration parameters
  std::vector<adas::core::PolynomialLaneBoundary> boundaries =
      adas::perception::detect_lanes(bev, config);

  // Compute binary image for visualization using config threshold
  cv::Mat gray, binary;
  if (bev.channels() == 3) {
    cv::cvtColor(bev, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = bev;
  }
  cv::threshold(gray, binary, config.white_pixel_threshold, 255,
                cv::THRESH_BINARY);

  cv::Mat debug_vis = bev.clone();

  if (boundaries.empty()) {
    std::cout << "No valid lane boundaries detected." << std::endl;
  } else {
    for (size_t i = 0; i < boundaries.size(); ++i) {
      const auto &lane = boundaries[i];

      // Draw fitted polynomial curve
      for (int y_coord = 0; y_coord < 480; y_coord += 2) {
        double x_coord = lane.evaluate(y_coord);
        if (x_coord >= 0 && x_coord < 640) {
          cv::circle(debug_vis, cv::Point(static_cast<int>(x_coord), y_coord),
                     2, cv::Scalar(0, 255, 0), -1);
        }
      }

      std::cout << "Fitted Polynomial [" << i << "]: y = " << lane.c0 << " + "
                << lane.c1 << "*x + " << lane.c2 << "*x^2 \n";
    }
  }

  cv::imshow("Original Frame", frame);
  cv::imshow("BEV Binary", binary);
  cv::imshow("Fitted Lane (BEV)", debug_vis);
  cv::waitKey(0);
  return 0;
}

} // namespace adas::perception
