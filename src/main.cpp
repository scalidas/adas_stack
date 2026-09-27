#include "config/config.hpp"
#include "perception/calibration.hpp"
#include "perception/lane_detection.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::string mode = "detect_lane";
  std::string config_path = "config/perception_config.json";
  std::string image_path_override = "";

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" || arg == "-c") {
      if (i + 1 < argc) {
        config_path = argv[++i];
      }
    } else if (arg == "calibrate" || arg == "detect_lane") {
      mode = arg;
    } else if (image_path_override.empty()) {
      image_path_override = arg;
    }
  }

  // Load configuration from JSON file
  adas::config::PerceptionConfig config = adas::config::load_config(config_path);

  // Use CLI image path override if provided
  if (!image_path_override.empty()) {
    config.image_path = image_path_override;
  }

  std::cout << "[Config Parameters]"
            << "\n  Image Path:              " << config.image_path
            << "\n  White Pixel Threshold:   " << config.white_pixel_threshold
            << "\n  RANSAC Iterations:       " << config.ransac_iterations
            << "\n  RANSAC Inlier Threshold: " << config.ransac_inlier_threshold << " px"
            << "\n  Min Inlier Count:        " << config.min_inlier_count << "\n\n";

  if (mode == "calibrate") {
    std::cout << "[Mode] Running Interactive Calibration on: " << config.image_path
              << std::endl;
    adas::calibration::interactive_calibration(config.image_path);
  } else if (mode == "detect_lane") {
    std::cout << "[Mode] Running Lane Detection on: " << config.image_path
              << std::endl;

    cv::Mat frame = cv::imread(config.image_path);
    if (frame.empty()) {
      std::cerr << "Failed to load image: " << config.image_path << std::endl;
      return -1;
    }
    cv::resize(frame, frame, cv::Size(640, 480));

    // Perspective transform to Birds-Eye-View (BEV)
    std::vector<cv::Point2f> src_pts = { {223.0f, 240.0f}, {391.0f, 241.0f}, {170.0f, 393.0f}, {444.0f, 394.0f}} ;

    // Target BEV points for 168mm (W) x 216mm (L) on a 640x480 canvas
    std::vector<cv::Point2f> dst_pts = {
        {220.0f, 143.0f}, // Top-Left     (matches {223.0f, 240.0f})
        {420.0f, 143.0f}, // Top-Right    (matches {391.0f, 241.0f})
        {220.0f, 400.0f}, // Bottom-Left  (matches {170.0f, 393.0f})
        {420.0f, 400.0f}  // Bottom-Right (matches {444.0f, 394.0f})
    };

    cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat bev;
    cv::warpPerspective(frame, bev, H, cv::Size(640, 480));

    // Run lane detection using loaded configuration parameters
    std::vector<adas::core::PolynomialLaneBoundary> boundaries =
        adas::perception::detect_lanes(bev, config);

    // Compute binary image for visualization using config threshold
    cv::Mat gray, binary;
    cv::cvtColor(bev, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, config.white_pixel_threshold, 255, cv::THRESH_BINARY);

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
  }

  return 0;
}