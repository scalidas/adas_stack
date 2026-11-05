#pragma once

#ifdef ADAS_WITH_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/core_types.hpp"
#include "yolo/inference_engine.hpp"

namespace adas::yolo {

// ---------------------------------------------------------------------------
// TensorRT logger — routes TRT diagnostics to stderr
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger {
   public:
    explicit TRTLogger(nvinfer1::ILogger::Severity min_severity =
                           nvinfer1::ILogger::Severity::kWARNING)
        : min_severity_(min_severity) {}

    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override;

   private:
    nvinfer1::ILogger::Severity min_severity_;
};

// ---------------------------------------------------------------------------
// TensorRT inference engine (YOLOv8 ONNX → TRT engine)
// ---------------------------------------------------------------------------
class TensorRTEngine : public InferenceEngine {
   public:
    TensorRTEngine() = default;
    explicit TensorRTEngine(const std::string& model_path,
                            float conf_threshold = 0.25f,
                            float nms_threshold = 0.45f);
    ~TensorRTEngine() override;

    // ---- InferenceEngine interface ----------------------------------------
    bool loadModel(const std::string& model_path) override;
    std::vector<adas::core::DetectedObject> infer(
        const cv::Mat& frame) override;
    bool runLiveDemo(const adas::config::PerceptionConfig& config) override;
    bool runLiveDemo(int webcam_id = 0,
                     const std::string& intrinsics_path = "") override;
    void setClasses(const std::vector<int>& classes) override;
    void setClasses(const std::vector<std::string>& classes) override;

    // ---- Extra configuration (mirrors OpenCVDNNEngine API) ----------------
    void setConfidenceThreshold(float conf_threshold);
    void setNMSThreshold(float nms_threshold);
    void setInputSize(int width, int height);

    void setCameraIntrinsics(double fx, double fy, double cx, double cy,
                             int calib_w = 1920, int calib_h = 1080);
    bool loadCameraIntrinsics(const std::string& intrinsics_path);
    bool loadExtrinsicCalibration(const std::string& extrinsics_path);
    void setGroundTransform(const cv::Mat& ground_transform);
    const cv::Mat& getGroundTransform() const;
    void setObjectRealHeight(double height);
    void setClassRealHeights(const std::map<int, double>& heights);

    // ---- Getters -----------------------------------------------------------
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
    const std::map<int, double>& getClassRealHeights() const;
    const std::vector<int>& getTargetClassIds() const;

   private:
    // ---- Build / serialise -------------------------------------------------
    /// Build a TRT engine from an ONNX file and write a .trt plan alongside it.
    bool buildEngineFromONNX(const std::string& onnx_path,
                             const std::string& plan_path);
    /// Load a pre-serialised .trt engine plan from disk.
    bool loadEnginePlan(const std::string& plan_path);

    // ---- Post-processing helpers -------------------------------------------
    /// Parse the raw network output tensor and return filtered detections.
    std::vector<adas::core::DetectedObject> postprocess(
        const float* output_data, int num_channels, int num_anchors,
        int frame_cols, int frame_rows);

    /// Scale camera intrinsics when the live frame resolution differs from the
    /// calibration resolution.
    void scaledIntrinsics(int frame_cols, int frame_rows, double& fx,
                          double& fy, double& cx, double& cy) const;

    // ---- TRT objects -------------------------------------------------------
    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // ---- CUDA buffers (device pointers, one per binding) -------------------
    std::vector<void*> device_buffers_;
    /// Host-side staging buffer for the output tensor.
    std::vector<float> host_output_buffer_;

    // ---- Network geometry --------------------------------------------------
    int input_width_{640};
    int input_height_{640};
    int input_binding_idx_{-1};
    int output_binding_idx_{-1};
    int output_num_channels_{0};  // e.g. 84 for 80-class COCO
    int output_num_anchors_{0};   // e.g. 8400

    // ---- Detection parameters ----------------------------------------------
    bool is_loaded_{false};
    float conf_threshold_{0.25f};
    float nms_threshold_{0.45f};

    // ---- Class lists -------------------------------------------------------
    std::vector<std::string> classes_;
    std::vector<int> target_class_ids_;

    // ---- Camera intrinsics & extrinsics ------------------------------------
    double fx_{1056.46136};
    double cx_{1017.33354};
    double fy_{1063.57938};
    double cy_{535.71139};
    int calib_w_{1920};
    int calib_h_{1080};
    bool has_calibration_{true};
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Mat ground_transform_;

    // ---- Physical object heights (metres) ----------------------------------
    double default_real_height_{1.5};
    std::map<int, double> class_real_heights_;
};

}  // namespace adas::yolo

using TensorRTEngine = adas::yolo::TensorRTEngine;

#endif  // ADAS_WITH_TENSORRT

