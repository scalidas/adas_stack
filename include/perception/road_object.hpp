#pragma once

#include "core/core_types.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace adas::perception {

class RoadObject {
public:
  // Constructor taking DetectedObject, physical object height in meters, and camera intrinsics parameters
  RoadObject(const adas::core::DetectedObject &detected_obj,
             double real_height_m = 1.5,
             double fy = 1063.57938,
             double fx = 1056.46136,
             double cx = 1017.33354,
             double cy = 535.71139);

  // Constructor taking DetectedObject, physical object height in meters, and OpenCV camera matrix
  RoadObject(const adas::core::DetectedObject &detected_obj,
             double real_height_m,
             const cv::Mat &camera_matrix);

  // Setters for intrinsic parameters and object dimensions
  void set_intrinsics(double fy, double fx, double cx, double cy);
  void set_intrinsics(const cv::Mat &camera_matrix);
  void set_real_height(double real_height_m);

  // Process method returning TrackedObject with 3D pose relative to camera
  adas::core::TrackedObject process() const;

private:
  adas::core::DetectedObject detected_obj_;
  double real_height_m_{1.5};
  double fy_{1063.57938};
  double fx_{1056.46136};
  double cx_{1017.33354};
  double cy_{535.71139};
};

// Demo function to test camera intrinsics and 3D pose estimation from config
int run_road_object_demo(const std::string &config_path);

} // namespace adas::perception
