#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


#include "core/core_types.hpp"

namespace adas::yolo {

class InferenceEngine {
public:
  virtual ~InferenceEngine() = default;
  virtual bool loadModel(const std::string &model_path) = 0;
  virtual std::vector<adas::core::DetectedObject>
  infer(const cv::Mat &frame) = 0;
  virtual bool runLiveDemo(int webcam_id = 0,
                           const std::string &intrinsics_path = "") = 0;
};

} // namespace adas::yolo

using InferenceEngine = adas::yolo::InferenceEngine;
