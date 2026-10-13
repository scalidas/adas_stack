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
  void setClasses(const std::vector<std::string> &classes);

  bool isLoaded() const;
  float getConfidenceThreshold() const;
  float getNMSThreshold() const;

private:
  cv::dnn::Net net_;
  bool is_loaded_{false};
  float conf_threshold_{0.25f};
  float nms_threshold_{0.45f};
  int input_width_{640};
  int input_height_{640};
  std::vector<std::string> classes_;
};

} // namespace adas::yolo

using OpenCVDNNEngine = adas::yolo::OpenCVDNNEngine;