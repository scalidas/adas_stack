#include "config/config.hpp"
#include "calibration/intrinsic_calibration.hpp"
#include "calibration/extrinsic_calibration.hpp"
#include "perception/lane_detection.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum mode {
  INTRINSIC_CALIBRATION,
  EXTRINSIC_CALIBRATION,
  DETECT_LANE_DEMO,
};

int main(int argc, char **argv) {
  //Default paths
  const std::string_view default_config_path = "config/perception_config.json";

  //Known camera extrinsic calibration
  // Perspective transform to Birds-Eye-View (BEV)
  const std::vector<cv::Point2f> src_pts = { 
    {223.0f, 240.0f}, 
    {391.0f, 241.0f}, 
    {170.0f, 393.0f}, 
    {444.0f, 394.0f}
  };

  // Target BEV points for 168mm (W) x 216mm (L) on a 640x480 canvas
  const std::vector<cv::Point2f> dst_pts = {
    {220.0f, 143.0f}, 
    {420.0f, 143.0f}, 
    {220.0f, 400.0f},
    {420.0f, 400.0f}
  };

  //Compute homography matrix
  const cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);

  // Parse command line arguments
  if (argc < 2) {
    std::cout << "Invalid usage: Specify mode of operation" << std::endl;
    return -1;
  }

  mode mode = INTRINSIC_CALIBRATION;
  std::string arg_mode = argv[1];

  //Set mode according to the passed in mode
  if (arg_mode == "intrinsic_calibration") {
    mode = INTRINSIC_CALIBRATION;
  } else if (arg_mode == "extrinsic_calibration") {
    mode = EXTRINSIC_CALIBRATION;
  } else if (arg_mode == "detect_lane_demo") {
    mode = DETECT_LANE_DEMO;
  } else {
    std::cout << "Invalid mode selection" << std::endl;
    return -1;
  }

  switch (mode) {
    case INTRINSIC_CALIBRATION: {
      //Expect config file to be passed in
      if (argc != 3) {
        std::cout << "Invalid number of arguments" << std::endl;
        return -1;
      }

      std::string config_file = argv[2];
      return adas::calibration::run_intrinsic_calibration(config_file);
    }
    case EXTRINSIC_CALIBRATION: {
      //Expect image file to be passed in
      if (argc != 3) {
        std::cout << "Invalid number of arguments" << std::endl;
        return -1;
      }

      std::string image_file = argv[2];
      return adas::calibration::run_extrinsic_calibration(image_file);
    }
    case DETECT_LANE_DEMO: {
      //Expect config file to be passed in
      if (argc != 3) {
        std::cout << "Invalid number of arguments" << std::endl;
        return -1;
      }

      std::string config_file = argv[2];
      adas::config::PerceptionConfig config = adas::config::load_config(config_file);
    }
  }

  return 0;
}