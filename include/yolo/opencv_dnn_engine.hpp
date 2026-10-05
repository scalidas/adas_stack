/*
Inference engine overridesshould abstract away the underlying runtime. Should consume ONNX model or 
tensorrt model and return bounding boxes in unified format
*/
#include <string>
#include <opencv2/opencv.hpp>

#include "core/core_types.hpp"
#include "yolo/inference_engine.hpp"

class OpenCVDNNEngine : InferenceEngine {
    ~OpenCVDNNEngine() = default;
    bool loadModel(const std::string& model_path) override;
    std::vector<adas::core::DetectedObject> infer(const cv::Mat& frame) override;
};