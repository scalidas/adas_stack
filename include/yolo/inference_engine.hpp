/*
Inference engine overridesshould abstract away the underlying runtime. Should consume ONNX model or 
tensorrt model and return bounding boxes in unified format
*/
#include <string>
#include <opencv2/opencv.hpp>

#include "core/core_types.hpp"

class InferenceEngine {
    virtual ~InferenceEngine() = default;
    virtual bool loadModel(const std::string& model_path) = 0;
    virtual std::vector<adas::core::DetectedObject> infer(const cv::Mat& frame) = 0;
};