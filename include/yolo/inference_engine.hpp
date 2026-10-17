#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "config/config.hpp"
#include "core/core_types.hpp"

namespace adas::yolo {

static const std::vector<std::string> coco_classes = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

class InferenceEngine {
   public:
    virtual ~InferenceEngine() = default;
    virtual bool loadModel(const std::string& model_path) = 0;
    virtual std::vector<adas::core::DetectedObject> infer(
        const cv::Mat& frame) = 0;

    virtual void setClasses(const std::vector<int> &classes) = 0;
    virtual void setClasses(const std::vector<std::string>& classes) = 0;

    virtual bool runLiveDemo(const adas::config::PerceptionConfig& config) = 0;

    virtual bool runLiveDemo(int webcam_id = 0, const std::string& intrinsics_path = "") = 0;
};

}  // namespace adas::yolo

using InferenceEngine = adas::yolo::InferenceEngine;
