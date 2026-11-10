#ifdef ADAS_WITH_TENSORRT

#include "yolo/tensorrt_engine.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "calibration/extrinsic_calibration.hpp"
#include "config/config.hpp"
#include "perception/road_object.hpp"
#include "yolo/inference_engine.hpp"

// ---------------------------------------------------------------------------
// Convenience macro — throws on CUDA error
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("[TensorRTEngine] CUDA ") + \
                                     cudaGetErrorString(_err) + " at " +   \
                                     __FILE__ + ":" +                      \
                                     std::to_string(__LINE__));             \
        }                                                                   \
    } while (false)

namespace adas::yolo {

// ---------------------------------------------------------------------------
// TRTLogger
// ---------------------------------------------------------------------------
void TRTLogger::log(nvinfer1::ILogger::Severity severity,
                    const char* msg) noexcept {
    if (severity > min_severity_) {
        return;
    }
    const char* prefix = "[TRT] ";
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << prefix << "ERROR: " << msg << "\n";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << prefix << "WARN: " << msg << "\n";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cout << prefix << "INFO: " << msg << "\n";
            break;
        default:
            std::cout << prefix << msg << "\n";
            break;
    }
}

// ---------------------------------------------------------------------------
// Constructors / Destructor
// ---------------------------------------------------------------------------
TensorRTEngine::TensorRTEngine(const std::string& model_path,
                               float conf_threshold, float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
    loadModel(model_path);
}

TensorRTEngine::~TensorRTEngine() {
    // Free CUDA device buffers
    for (void* ptr : device_buffers_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    device_buffers_.clear();
}

// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------
bool TensorRTEngine::loadModel(const std::string& model_path) {
    is_loaded_ = false;

    if (model_path.empty()) {
        std::cerr << "[TensorRTEngine] Model path is empty.\n";
        return false;
    }

    // Determine plan path: same location as ONNX but with .trt extension
    std::filesystem::path onnx_path(model_path);
    std::filesystem::path plan_path = onnx_path;
    plan_path.replace_extension(".trt");

    // Initialise runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "[TensorRTEngine] Failed to create TRT runtime.\n";
        return false;
    }

    // Try to load a cached plan first; fall back to building from ONNX
    if (std::filesystem::exists(plan_path)) {
        std::cout << "[TensorRTEngine] Loading cached engine plan: "
                  << plan_path << "\n";
        if (!loadEnginePlan(plan_path.string())) {
            std::cerr << "[TensorRTEngine] Cached plan load failed; rebuilding "
                         "from ONNX.\n";
        }
    }

    if (!engine_) {
        std::cout << "[TensorRTEngine] Building engine from ONNX: " << onnx_path
                  << "\n";
        if (!buildEngineFromONNX(onnx_path.string(), plan_path.string())) {
            std::cerr
                << "[TensorRTEngine] Failed to build engine from ONNX.\n";
            return false;
        }
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[TensorRTEngine] Failed to create execution context.\n";
        return false;
    }

    // ---------------------------------------------------------------------------
    // Discover input / output bindings
    // ---------------------------------------------------------------------------
    const int num_bindings = engine_->getNbIOTensors();
    device_buffers_.resize(static_cast<size_t>(num_bindings), nullptr);

    for (int i = 0; i < num_bindings; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        nvinfer1::Dims dims = engine_->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            // Expected: [1, 3, H, W]
            input_binding_idx_ = i;
            if (dims.nbDims == 4) {
                input_height_ = dims.d[2];
                input_width_ = dims.d[3];
            }
        } else {
            // Expected: [1, num_channels, num_anchors]
            output_binding_idx_ = i;
            if (dims.nbDims == 3) {
                output_num_channels_ = dims.d[1];
                output_num_anchors_ = dims.d[2];
            }
        }

        // Compute element count and allocate device buffer
        size_t num_elements = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            num_elements *= static_cast<size_t>(dims.d[d]);
        }
        CUDA_CHECK(cudaMalloc(&device_buffers_[static_cast<size_t>(i)],
                              num_elements * sizeof(float)));
    }

    // Allocate host-side output staging buffer
    host_output_buffer_.resize(
        static_cast<size_t>(output_num_channels_) *
        static_cast<size_t>(output_num_anchors_));

    // Wire tensor addresses into the execution context
    for (int i = 0; i < num_bindings; ++i) {
        const char* name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, device_buffers_[static_cast<size_t>(i)]);
    }

    is_loaded_ = true;
    std::cout << "[TensorRTEngine] Engine ready. Input: " << input_width_ << "x"
              << input_height_ << "  Output: [" << output_num_channels_ << " x "
              << output_num_anchors_ << "]\n";
    return true;
}

// ---------------------------------------------------------------------------
// buildEngineFromONNX
// ---------------------------------------------------------------------------
bool TensorRTEngine::buildEngineFromONNX(const std::string& onnx_path,
                                         const std::string& plan_path) {
    // Builder
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        return false;
    }

    // Network
    const uint32_t explicit_batch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }

    // ONNX parser
    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        return false;
    }

    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TensorRTEngine] ONNX parse failed for: " << onnx_path
                  << "\n";
        return false;
    }

    // Build configuration
    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Enable FP16 if the GPU supports it
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TensorRTEngine] FP16 precision enabled.\n";
    }

    // Serialise and write plan to disk
    std::unique_ptr<nvinfer1::IHostMemory> serialised(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialised || serialised->size() == 0) {
        std::cerr << "[TensorRTEngine] Engine serialisation failed.\n";
        return false;
    }

    {
        std::ofstream plan_file(plan_path, std::ios::binary);
        if (!plan_file) {
            std::cerr << "[TensorRTEngine] Cannot write plan: " << plan_path
                      << "\n";
            return false;
        }
        plan_file.write(static_cast<const char*>(serialised->data()),
                        static_cast<std::streamsize>(serialised->size()));
        std::cout << "[TensorRTEngine] Saved engine plan: " << plan_path
                  << "\n";
    }

    // Deserialise into engine_
    engine_.reset(runtime_->deserializeCudaEngine(serialised->data(),
                                                  serialised->size()));
    return engine_ != nullptr;
}

// ---------------------------------------------------------------------------
// loadEnginePlan
// ---------------------------------------------------------------------------
bool TensorRTEngine::loadEnginePlan(const std::string& plan_path) {
    std::ifstream file(plan_path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) {
        return false;
    }

    engine_.reset(
        runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    return engine_ != nullptr;
}

// ---------------------------------------------------------------------------
// infer  (with per-stage timing, printed every 30 frames)
// ---------------------------------------------------------------------------
std::vector<adas::core::DetectedObject> TensorRTEngine::infer(
    const cv::Mat& frame) {
    std::vector<adas::core::DetectedObject> detections;

    if (!is_loaded_ || frame.empty()) {
        return detections;
    }

    // Helpers -----------------------------------------------------------------
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b) -> double {
        return std::chrono::duration_cast<Ms>(b - a).count();
    };

    static uint32_t frame_idx = 0;
    constexpr uint32_t kLogInterval = 30; // print timing summary every N frames

    try {
        // ---- 1. Pre-process: resize -> float [0,1] -> CHW layout -----------
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_width_, input_height_));

        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // Build a CHW blob (planar layout)
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);

        const size_t channel_size =
            static_cast<size_t>(input_width_) * static_cast<size_t>(input_height_);
        const size_t input_bytes = 3 * channel_size * sizeof(float);

        std::vector<float> input_blob(3 * channel_size);
        for (int c = 0; c < 3; ++c) {
            // OpenCV channels are BGR; copy in BGR order (ONNX export from
            // Ultralytics already handles the colour swap internally)
            std::memcpy(input_blob.data() + static_cast<size_t>(c) * channel_size,
                        channels[c].ptr<float>(),
                        channel_size * sizeof(float));
        }

        // ---- 2. Host -> Device copy ----------------------------------------
        CUDA_CHECK(cudaMemcpy(device_buffers_[static_cast<size_t>(input_binding_idx_)],
                              input_blob.data(), input_bytes,
                              cudaMemcpyHostToDevice));

        // ---- 3. GPU inference ----------------------------------------------
        if (!context_->executeV2(device_buffers_.data())) {
            std::cerr << "[TensorRTEngine] executeV2 failed.\n";
            return detections;
        }
        // Synchronise so the timer reflects actual GPU work, not just kernel launch
        cudaDeviceSynchronize();


        // ---- 4. Device -> Host copy ----------------------------------------
        const size_t output_bytes =
            static_cast<size_t>(output_num_channels_) *
            static_cast<size_t>(output_num_anchors_) * sizeof(float);
        CUDA_CHECK(
            cudaMemcpy(host_output_buffer_.data(),
                       device_buffers_[static_cast<size_t>(output_binding_idx_)],
                       output_bytes, cudaMemcpyDeviceToHost));

        auto t4 = Clock::now();

        // ---- 5. Post-process (CPU NMS + depth estimation) ------------------
        detections = postprocess(host_output_buffer_.data(),
                                 output_num_channels_, output_num_anchors_,
                                 frame.cols, frame.rows);



    } catch (const std::exception& e) {
        std::cerr << "[TensorRTEngine] Exception during infer: " << e.what()
                  << "\n";
    }

    return detections;
}
// ---------------------------------------------------------------------------
// postprocess — YOLOv8 output layout: [num_channels, num_anchors]
//               channels 0-3: cx, cy, w, h (in input-space pixels)
//               channels 4..N: per-class scores
// ---------------------------------------------------------------------------
std::vector<adas::core::DetectedObject> TensorRTEngine::postprocess(
    const float* data, int num_channels, int num_anchors, int frame_cols,
    int frame_rows) {

    std::vector<adas::core::DetectedObject> detections;

    const float x_scale = static_cast<float>(frame_cols) / input_width_;
    const float y_scale = static_cast<float>(frame_rows) / input_height_;

    std::vector<cv::Rect2d> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int a = 0; a < num_anchors; ++a) {
        // Stride through the [channels, anchors] layout
        const float cx = data[0 * num_anchors + a];
        const float cy = data[1 * num_anchors + a];
        const float w  = data[2 * num_anchors + a];
        const float h  = data[3 * num_anchors + a];

        float max_score = 0.0f;
        int best_class  = -1;
        for (int c = 4; c < num_channels; ++c) {
            float score = data[c * num_anchors + a];
            if (score > max_score) {
                max_score  = score;
                best_class = c - 4;
            }
        }

        if (max_score < conf_threshold_) {
            continue;
        }

        const float left  = (cx - 0.5f * w) * x_scale;
        const float top   = (cy - 0.5f * h) * y_scale;
        const float width = w * x_scale;
        const float height = h * y_scale;

        bboxes.emplace_back(left, top, width, height);
        confidences.push_back(max_score);
        class_ids.push_back(best_class);
    }

    if (bboxes.empty()) {
        return detections;
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, confidences, conf_threshold_, nms_threshold_,
                      nms_indices);

    double cur_fx, cur_fy, cur_cx, cur_cy;
    scaledIntrinsics(frame_cols, frame_rows, cur_fx, cur_fy, cur_cx, cur_cy);

    for (int idx : nms_indices) {
        const int cid = class_ids[idx];

        // Filter to requested classes (if any)
        if (!target_class_ids_.empty()) {
            if (std::find(target_class_ids_.begin(), target_class_ids_.end(),
                          cid) == target_class_ids_.end()) {
                continue;
            }
        }

        adas::core::DetectedObject obj;
        obj.bounding_box = cv::Rect2f(static_cast<float>(bboxes[idx].x),
                                      static_cast<float>(bboxes[idx].y),
                                      static_cast<float>(bboxes[idx].width),
                                      static_cast<float>(bboxes[idx].height));
        obj.confidence = confidences[idx];
        obj.class_id   = cid;

        // Label
        if (cid >= 0 && cid < static_cast<int>(coco_classes.size())) {
            obj.label = coco_classes[cid];
        } else {
            obj.label = "class_" + std::to_string(cid);
        }

        // Physical height for depth estimation
        double real_height = default_real_height_;
        if (class_real_heights_.count(cid)) {
            real_height = class_real_heights_.at(cid);
        } else {
            switch (cid) {
                case 0:  real_height = 1.70; break;  // person
                case 1:  real_height = 1.00; break;  // bicycle
                case 2:  real_height = 1.50; break;  // car
                case 3:  real_height = 1.20; break;  // motorcycle
                case 5:  real_height = 3.20; break;  // bus
                case 7:  real_height = 3.00; break;  // truck
                case 9:  real_height = 0.80; break;  // traffic light
                case 11: real_height = 0.75; break;  // stop sign
                case 39: real_height = 0.25; break;  // bottle
                case 41: real_height = 0.10; break;  // cup
                default: break;
            }
        }

        if (obj.bounding_box.height > 0.0f) {
            adas::perception::RoadObject road_obj(obj, real_height, cur_fy,
                                                  cur_fx, cur_cx, cur_cy,
                                                  ground_transform_);
            obj.position = road_obj.process().position;
        }

        detections.push_back(obj);
    }

    return detections;
}

// ---------------------------------------------------------------------------
// scaledIntrinsics
// ---------------------------------------------------------------------------
void TensorRTEngine::scaledIntrinsics(int frame_cols, int frame_rows,
                                      double& out_fx, double& out_fy,
                                      double& out_cx, double& out_cy) const {
    if (calib_w_ > 0 && calib_h_ > 0 &&
        (frame_cols != calib_w_ || frame_rows != calib_h_)) {
        const double sx = static_cast<double>(frame_cols) / calib_w_;
        const double sy = static_cast<double>(frame_rows) / calib_h_;
        out_fx = fx_ * sx;
        out_cx = cx_ * sx;
        out_fy = fy_ * sy;
        out_cy = cy_ * sy;
    } else if (!has_calibration_) {
        out_fx = 0.8 * frame_cols;
        out_fy = 0.8 * frame_cols;
        out_cx = frame_cols / 2.0;
        out_cy = frame_rows / 2.0;
    } else {
        out_fx = fx_;
        out_fy = fy_;
        out_cx = cx_;
        out_cy = cy_;
    }
}

// ---------------------------------------------------------------------------
// runLiveDemo (PerceptionConfig overload)
// ---------------------------------------------------------------------------
bool TensorRTEngine::runLiveDemo(
    const adas::config::PerceptionConfig& config) {
    cv::VideoCapture cap(config.webcam_id);
    if (!cap.isOpened()) {
        std::cerr << "[TensorRTEngine] Cannot open webcam "
                  << config.webcam_id << "\n";
        return false;
    }

    default_real_height_ = config.object_real_height;
    class_real_heights_  = config.class_real_heights;
    if (!config.target_class_ids.empty()) {
        setClasses(config.target_class_ids);
    }

    if (!config.intrinsics_path.empty()) {
        if (!loadCameraIntrinsics(config.intrinsics_path)) {
            std::cerr << "[TensorRTEngine] Warning: Falling back to default "
                         "intrinsics.\n";
        }
    } else {
        has_calibration_ = false;
    }

    if (!config.extrinsics_path.empty()) {
        loadExtrinsicCalibration(config.extrinsics_path);
    }

    if (calib_w_ > 0 && calib_h_ > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  calib_w_);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, calib_h_);
    }

    // -------------------------------------------------------------------------
    // Precompute undistortion remap maps once so we call the fast cv::remap()
    // each frame instead of the slow cv::undistort() (which recomputes maps
    // internally on every single call -- the dominant cost at 1080p on CPU).
    // -------------------------------------------------------------------------
    cv::Mat remap_map1, remap_map2;
    bool has_remap = false;

    if (!camera_matrix_.empty() && !dist_coeffs_.empty()) {
        // Grab the actual capture resolution (may differ from calib resolution)
        cv::Mat probe_frame;
        cap >> probe_frame;
        if (!probe_frame.empty()) {
            const int fw = probe_frame.cols;
            const int fh = probe_frame.rows;

            // Scale camera matrix to match actual capture resolution
            cv::Mat K = camera_matrix_.clone();
            if (fw != calib_w_ || fh != calib_h_) {
                const double sx = static_cast<double>(fw) / calib_w_;
                const double sy = static_cast<double>(fh) / calib_h_;
                K.at<double>(0, 0) *= sx;  // fx
                K.at<double>(0, 2) *= sx;  // cx
                K.at<double>(1, 1) *= sy;  // fy
                K.at<double>(1, 2) *= sy;  // cy
            }

            cv::initUndistortRectifyMap(
                K, dist_coeffs_,
                cv::Mat(),          // no rectification rotation
                K,                  // same output camera matrix
                cv::Size(fw, fh),
                CV_16SC2,           // fixed-point map — fastest for remap()
                remap_map1, remap_map2);

            has_remap = true;
            std::printf("[TensorRTEngine] Precomputed undistort remap maps "
                        "for %dx%d frames.\n", fw, fh);

            // Push the probe frame back into processing below
            // (we'll handle it as the first loop iteration)
            // Re-open is simpler than buffering; re-read on first iteration.
        }
    }

    std::printf("[TensorRTEngine] Starting live demo on webcam %d. "
                "Press 'q' or ESC to exit.\n", config.webcam_id);

    const std::string window_name = "ADAS TensorRT Live Demo";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    // Loop-level timing -------------------------------------------------------
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b) -> double {
        return std::chrono::duration_cast<Ms>(b - a).count();
    };

    static uint32_t loop_idx = 0;
    constexpr uint32_t kLoopLogInterval = 30;

    cv::Mat raw_frame, frame;
    while (true) {
        auto tL0 = Clock::now();

        // ---- Capture --------------------------------------------------------
        cap >> raw_frame;
        if (raw_frame.empty()) {
            std::cerr << "[TensorRTEngine] Warning: empty frame captured.\n";
            break;
        }

        auto tL1 = Clock::now();

        // ---- Undistort (fast remap path) ------------------------------------
        if (has_remap) {
            cv::remap(raw_frame, frame, remap_map1, remap_map2,
                      cv::INTER_LINEAR);
        } else {
            frame = raw_frame;
        }

        auto tL2 = Clock::now();

        // ---- TRT inference (internally timed every 30 frames) ---------------
        auto detections = infer(frame);

        auto tL3 = Clock::now();

        // ---- Draw overlays --------------------------------------------------
        size_t rendered_count = 0;
        for (const auto& obj : detections) {
            if (!config.target_class_ids.empty()) {
                if (std::find(config.target_class_ids.begin(),
                              config.target_class_ids.end(),
                              obj.class_id) == config.target_class_ids.end()) {
                    continue;
                }
            }

            cv::Rect bbox_int(static_cast<int>(obj.bounding_box.x),
                              static_cast<int>(obj.bounding_box.y),
                              static_cast<int>(obj.bounding_box.width),
                              static_cast<int>(obj.bounding_box.height));
            cv::rectangle(frame, bbox_int, cv::Scalar(0, 128, 255), 2);

            std::string line1 = cv::format(
                "%s %.2f | Fwd: %.2fm [TRT]",
                obj.label.c_str(), obj.confidence, obj.position.x);
            std::string line2 = cv::format(
                "Ground: [X=%.2f, Y=%.2f, Z=%.2f]m",
                obj.position.x, obj.position.y, obj.position.z);

            const int font  = cv::FONT_HERSHEY_SIMPLEX;
            const double sc = 0.45;
            const int th    = 1;
            int b1 = 0, b2 = 0;
            cv::Size s1 = cv::getTextSize(line1, font, sc, th, &b1);
            cv::Size s2 = cv::getTextSize(line2, font, sc, th, &b2);
            int bw = std::max(s1.width, s2.width) + 8;
            int bh = s1.height + s2.height + 8;
            int bx = std::clamp(bbox_int.x, 0, std::max(0, frame.cols - bw));
            int by = std::max(bh + 2, bbox_int.y);

            cv::rectangle(frame, cv::Point(bx, by - bh),
                          cv::Point(bx + bw, by),
                          cv::Scalar(25, 25, 25), cv::FILLED);
            cv::putText(frame, line1, cv::Point(bx + 4, by - s2.height - 4),
                        font, sc, cv::Scalar(0, 128, 255), th);
            cv::putText(frame, line2, cv::Point(bx + 4, by - 2),
                        font, sc, cv::Scalar(0, 255, 255), th);

            ++rendered_count;
        }

        std::string hud = cv::format(
            "Tracked: %zu (of %zu) | TensorRT | 'q'/ESC to exit",
            rendered_count, detections.size());
        cv::putText(frame, hud, cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(0, 128, 255), 2);

        auto tL4 = Clock::now();

        // ---- Display --------------------------------------------------------
        cv::imshow(window_name, frame);
        const int key = cv::waitKey(1);

        auto tL5 = Clock::now();

        // ---- Loop timing log ------------------------------------------------
        ++loop_idx;
        if (loop_idx % kLoopLogInterval == 0) {
            const double ms_cap     = elapsed_ms(tL0, tL1);
            const double ms_undist  = elapsed_ms(tL1, tL2);
            const double ms_infer   = elapsed_ms(tL2, tL3);
            const double ms_draw    = elapsed_ms(tL3, tL4);
            const double ms_display = elapsed_ms(tL4, tL5);
            const double ms_total   = elapsed_ms(tL0, tL5);

            std::printf(
                "[LOOP timing #%u]  cap=%.1fms  undist=%.1fms  infer=%.1fms"
                "  draw=%.1fms  display=%.1fms  | total=%.1fms (%.1f fps)\n",
                loop_idx,
                ms_cap, ms_undist, ms_infer, ms_draw, ms_display,
                ms_total, 1000.0 / ms_total);
            std::fflush(stdout);
        }

        if (key == 27 || key == 'q' || key == 'Q') {
            std::printf("[TensorRTEngine] Live demo stopped by user.\n");
            break;
        }
    }

    cv::destroyWindow(window_name);
    return true;
}
bool TensorRTEngine::runLiveDemo(int webcam_id,
                                 const std::string& intrinsics_path) {
    adas::config::PerceptionConfig config;
    config.webcam_id      = webcam_id;
    config.intrinsics_path = intrinsics_path;
    return runLiveDemo(config);
}

// ---------------------------------------------------------------------------
// setClasses
// ---------------------------------------------------------------------------
void TensorRTEngine::setClasses(const std::vector<std::string>& classes) {
    classes_ = classes;
    target_class_ids_.clear();
    for (const auto& name : classes) {
        auto it = std::find(coco_classes.begin(), coco_classes.end(), name);
        if (it != coco_classes.end()) {
            target_class_ids_.push_back(
                static_cast<int>(std::distance(coco_classes.begin(), it)));
        }
    }
}

void TensorRTEngine::setClasses(const std::vector<int>& classes) {
    target_class_ids_ = classes;
    classes_.clear();
    for (int idx : classes) {
        if (idx >= 0 && idx < static_cast<int>(coco_classes.size())) {
            classes_.push_back(coco_classes[idx]);
        }
    }
}

// ---------------------------------------------------------------------------
// Camera calibration helpers
// ---------------------------------------------------------------------------
void TensorRTEngine::setCameraIntrinsics(double fx, double fy, double cx,
                                         double cy, int calib_w, int calib_h) {
    fx_ = fx;  fy_ = fy;  cx_ = cx;  cy_ = cy;
    calib_w_ = calib_w;  calib_h_ = calib_h;
    has_calibration_ = true;
    camera_matrix_ =
        (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
}

bool TensorRTEngine::loadCameraIntrinsics(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[TensorRTEngine] Cannot open intrinsics: " << path
                  << "\n";
        return false;
    }
    fs["camera_matrix"]            >> camera_matrix_;
    fs["distortion_coefficients"]  >> dist_coeffs_;
    fs["image_width"]              >> calib_w_;
    fs["image_height"]             >> calib_h_;

    if (!camera_matrix_.empty() && camera_matrix_.rows >= 3 &&
        camera_matrix_.cols >= 3) {
        fx_ = camera_matrix_.at<double>(0, 0);
        cx_ = camera_matrix_.at<double>(0, 2);
        fy_ = camera_matrix_.at<double>(1, 1);
        cy_ = camera_matrix_.at<double>(1, 2);
        has_calibration_ = true;
        std::cout << "[TensorRTEngine] Loaded intrinsics from: " << path
                  << "\n";
        return true;
    }
    std::cerr << "[TensorRTEngine] Bad camera_matrix in: " << path << "\n";
    return false;
}

bool TensorRTEngine::loadExtrinsicCalibration(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    return adas::calibration::compute_camera_to_ground_transform(
        path, camera_matrix_, dist_coeffs_, ground_transform_,
        calib_w_, calib_h_);
}

void TensorRTEngine::setGroundTransform(const cv::Mat& t) {
    ground_transform_ = t;
}
const cv::Mat& TensorRTEngine::getGroundTransform() const {
    return ground_transform_;
}
void TensorRTEngine::setObjectRealHeight(double h) {
    default_real_height_ = h;
}
void TensorRTEngine::setClassRealHeights(const std::map<int, double>& h) {
    class_real_heights_ = h;
}

// ---------------------------------------------------------------------------
// Simple setters / getters
// ---------------------------------------------------------------------------
void TensorRTEngine::setConfidenceThreshold(float t) { conf_threshold_ = t; }
void TensorRTEngine::setNMSThreshold(float t)        { nms_threshold_  = t; }
void TensorRTEngine::setInputSize(int w, int h)      { input_width_ = w; input_height_ = h; }

bool  TensorRTEngine::isLoaded()              const { return is_loaded_;          }
float TensorRTEngine::getConfidenceThreshold()const { return conf_threshold_;     }
float TensorRTEngine::getNMSThreshold()       const { return nms_threshold_;      }
double TensorRTEngine::getFx()                const { return fx_;                 }
double TensorRTEngine::getFy()                const { return fy_;                 }
double TensorRTEngine::getCx()                const { return cx_;                 }
double TensorRTEngine::getCy()                const { return cy_;                 }
int   TensorRTEngine::getCalibWidth()         const { return calib_w_;            }
int   TensorRTEngine::getCalibHeight()        const { return calib_h_;            }
double TensorRTEngine::getObjectRealHeight()  const { return default_real_height_;}
const std::map<int, double>& TensorRTEngine::getClassRealHeights() const {
    return class_real_heights_;
}
const std::vector<int>& TensorRTEngine::getTargetClassIds() const {
    return target_class_ids_;
}

}  // namespace adas::yolo

#endif  // ADAS_WITH_TENSORRT

