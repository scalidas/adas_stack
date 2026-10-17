#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "core/core_types.hpp"
#include "yolo/inference_engine.hpp"

namespace adas::yolo {

class OpenCVDNNEngine : public InferenceEngine {
public:
  OpenCVDNNEngine() = default;
  explicit OpenCVDNNEngine(const std::string &model_path,
                           float conf_threshold = 0.25f,
                           float nms_threshold = 0.45f);
  virtual ~OpenCVDNNEngine() override = default;

  bool loadModel(const std::string &model_path) override;
  std::vector<adas::core::DetectedObject>
  infer(const cv::Mat &frame) override;
  bool runLiveDemo(const adas::config::PerceptionConfig &config) override;
  bool runLiveDemo(int webcam_id = 0,
                   const std::string &intrinsics_path = "") override;

  void setConfidenceThreshold(float conf_threshold);
  void setNMSThreshold(float nms_threshold);
  void setInputSize(int width, int height);

  void setClasses(const std::vector<int> &classes) override;
  void setClasses(const std::vector<std::string> &classes) override;

  void setCameraIntrinsics(double fx, double fy, double cx, double cy,
                           int calib_w = 1920, int calib_h = 1080);
  bool loadCameraIntrinsics(const std::string &intrinsics_path);
  void setObjectRealHeight(double height);
  void setClassRealHeights(const std::map<int, double> &heights);

  bool isLoaded() const;
  float getConfidenceThreshold() const;
  float getNMSThreshold() const;
  double getFx() const;
  double getFy() const;
  double getCx() const;
  double getCy() const;
  int getCalibWidth() const;
  int getCalibHeight() const;
  double getObjectRealHeight() const;
  const std::map<int, double> &getClassRealHeights() const;
  const std::vector<int> &getTargetClassIds() const;

private:
  cv::dnn::Net net_;
  bool is_loaded_{false};
  float conf_threshold_{0.25f};
  float nms_threshold_{0.45f};
  int input_width_{640};
  int input_height_{640};
  std::vector<std::string> classes_;
  std::vector<int> target_class_ids_;

  // Camera calibration & intrinsics for 3D deprojection
  double fx_{1056.46136};
  double cx_{1017.33354};
  double fy_{1063.57938};
  double cy_{535.71139};
  int calib_w_{1920};
  int calib_h_{1080};
  bool has_calibration_{true};
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // Object real physical height configuration
  double default_real_height_{1.5};
  std::map<int, double> class_real_heights_;
};

} // namespace adas::yolo

using OpenCVDNNEngine = adas::yolo::OpenCVDNNEngine;
