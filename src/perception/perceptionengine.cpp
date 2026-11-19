#include <cstdio>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "calibration/extrinsic_calibration.hpp"
#include "config/config.hpp"
#include "controller/stanley_controller.hpp"
#include "core/core_types.hpp"
#include "perception/lane_detection.hpp"
#include "perception/perceptionengine.hpp"
#include "planning/planner.hpp"
#include "streaming/websocket_streamer.hpp"
#include "yolo/inference_engine.hpp"
#include "yolo/opencv_dnn_engine.hpp"

#ifdef ADAS_WITH_TENSORRT
#include "yolo/tensorrt_engine.hpp"
#endif  // ADAS_WITH_TENSORRT

static bool is_left(const adas::core::PolynomialLaneBoundary& a,
                    const adas::core::PolynomialLaneBoundary& b) {
    // In BEV pixel coords, evaluate at row 300; smaller x (column) is the left boundary
    return a.evaluate(300.0) < b.evaluate(300.0);
}

static adas::core::PolynomialLaneBoundary computeCenterline(
    const adas::core::PolynomialLaneBoundary& left,
    const adas::core::PolynomialLaneBoundary& right) {
    return adas::core::PolynomialLaneBoundary{.c0 = 0.5 * (left.c0 + right.c0),
                                              .c1 = 0.5 * (left.c1 + right.c1),
                                              .c2 = 0.5 * (left.c2 + right.c2)};
}

namespace adas::perception {

PerceptionEngine::PerceptionEngine() {
    frame_counter = 0;
}

PerceptionEngine::~PerceptionEngine() {
}

PerceptionEngine::PerceptionEngine(const adas::config::PerceptionConfig& cfg) {
    config = cfg;

    // Select inference backend based on config.engine_type
    // Supported values: "opencv" (default), "tensorrt" / "trt" (requires
    // the ADAS_WITH_TENSORRT build flag).
#ifdef ADAS_WITH_TENSORRT
    if (config.engine_type == "tensorrt") {
        auto* trt_engine = new adas::yolo::TensorRTEngine(config.model_path,
                                                          0.25f, 0.45f);
        inference_engine.reset(trt_engine);

        if (!config.intrinsics_path.empty()) {
            trt_engine->loadCameraIntrinsics(config.intrinsics_path);
        }
        if (!config.extrinsics_path.empty()) {
            trt_engine->loadExtrinsicCalibration(config.extrinsics_path);
        }
        trt_engine->setObjectRealHeight(config.object_real_height);
        if (!config.class_real_heights.empty()) {
            trt_engine->setClassRealHeights(config.class_real_heights);
        }
        std::cout << "[PerceptionEngine] Using TensorRT inference backend.\n";
    } else
#endif  // ADAS_WITH_TENSORRT
    {
        if (config.engine_type != "opencv" && !config.engine_type.empty()) {
            std::cerr << "[PerceptionEngine] Unknown engine_type \""
                      << config.engine_type
                      << "\". Falling back to OpenCV DNN.\n";
        }
        inference_engine =
            std::make_unique<adas::yolo::OpenCVDNNEngine>(config.model_path);

        if (auto* dnn_engine = dynamic_cast<adas::yolo::OpenCVDNNEngine*>(
                inference_engine.get())) {
            if (!config.intrinsics_path.empty()) {
                dnn_engine->loadCameraIntrinsics(config.intrinsics_path);
            }
            if (!config.extrinsics_path.empty()) {
                dnn_engine->loadExtrinsicCalibration(config.extrinsics_path);
            }
            dnn_engine->setObjectRealHeight(config.object_real_height);
            if (!config.class_real_heights.empty()) {
                dnn_engine->setClassRealHeights(config.class_real_heights);
            }
        }
        std::cout << "[PerceptionEngine] Using OpenCV DNN inference backend.\n";
    }

    #ifdef ADAS_ENABLE_MCU_LINK
    mcu_link = mcu_link::McuLink(cfg);
    #endif

    if (!config.target_class_ids.empty()) {
        inference_engine->setClasses(config.target_class_ids);
        for (int cid : config.target_class_ids) {
            if (cid >= 0 &&
                cid < static_cast<int>(adas::yolo::coco_classes.size())) {
                target_labels.insert(adas::yolo::coco_classes[cid]);
            }
        }
    }

    if (config.extrinsics_path.empty() ||
        !adas::calibration::load_extrinsic_calibration(config.extrinsics_path,
                                                       ipm) ||
        ipm.empty()) {
        const std::vector<cv::Point2f> src_pts = {{223.0f, 240.0f},
                                                  {391.0f, 241.0f},
                                                  {170.0f, 393.0f},
                                                  {444.0f, 394.0f}};
        const std::vector<cv::Point2f> dst_pts = {{220.0f, 143.0f},
                                                  {420.0f, 143.0f},
                                                  {220.0f, 400.0f},
                                                  {420.0f, 400.0f}};
        ipm = cv::getPerspectiveTransform(src_pts, dst_pts);
    }

    frame_counter = 0;
}

std::optional<adas::core::Lane> PerceptionEngine::process_lanes(
    const cv::Mat& frame) {
    //Project to BEV
    cv::Mat bev;
    cv::warpPerspective(frame, bev, ipm, cv::Size(640, 480));

    //Get the polynomial lane boundaries
    std::vector<adas::core::PolynomialLaneBoundary> lane_boundaries =
        detect_lanes(bev, config);

    //Expect two lanes for now
    if (lane_boundaries.size() != 2) {
        return std::nullopt;
    }

    //Construct lane model in BEV pixel coordinates
    adas::core::Lane lane;
    lane.lane_id = 1;

    //Left boundary is the one with smaller x in BEV
    lane.left_boundary = is_left(lane_boundaries[0], lane_boundaries[1])
                             ? lane_boundaries[0]
                             : lane_boundaries[1];
    lane.right_boundary = is_left(lane_boundaries[0], lane_boundaries[1])
                              ? lane_boundaries[1]
                              : lane_boundaries[0];

    lane.centerline =
        computeCenterline(lane.left_boundary, lane.right_boundary);

    constexpr double METERS_PER_BEV_PIXEL = 0.00084;
    double width_px =
        lane.right_boundary.evaluate(300.0) - lane.left_boundary.evaluate(300.0);
    if (width_px <= 0.0) {
        width_px = lane.right_boundary.c0 - lane.left_boundary.c0;
    }
    lane.lane_width = std::abs(width_px) * METERS_PER_BEV_PIXEL;

    return lane;
}

std::vector<adas::core::TrackedObject> PerceptionEngine::process_obstacles(
    const cv::Mat& frame) {
    std::vector<adas::core::DetectedObject> detections =
        inference_engine->infer(frame);

    std::vector<adas::core::TrackedObject> result;
    result.reserve(detections.size());

    for (auto detection : detections) {
        if (!config.target_class_ids.empty()) {
            if (std::find(config.target_class_ids.begin(),
                          config.target_class_ids.end(), detection.class_id) ==
                config.target_class_ids.end()) {
                continue;
            }
        }

        //Later on, we will need to run bytetrack to maintain the tracked objects. Now, we will
        //just instantiate new tracked objects

        adas::core::TrackedObject tracked_obj;
        tracked_obj.id = 0;
        tracked_obj.label = detection.label;
        tracked_obj.bounding_box = detection.bounding_box;
        tracked_obj.position = detection.position;

        //TODO: Set this based on label
        tracked_obj.length = 0.1;
        tracked_obj.width = 0.1;
        tracked_obj.height = 0.26;

        tracked_obj.confidence = detection.confidence;

        result.push_back(tracked_obj);
    }

    return result;
}


static double s_last_lanes_ms = 0.0;
static double s_last_yolo_ms = 0.0;

adas::core::WorldModel PerceptionEngine::process_frame(const cv::Mat& frame) {
    std::optional<adas::core::Lane> lane;
    std::vector<adas::core::TrackedObject> detected_objects;

    {
        // worker writes to its own dedicated stack variable
        std::jthread worker(
            [&lane, &frame, this] {
                auto t0 = std::chrono::steady_clock::now();
                lane = process_lanes(frame);
                auto t1 = std::chrono::steady_clock::now();
                s_last_lanes_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
            });

        // Main thread computes the other field locally
        auto ty0 = std::chrono::steady_clock::now();
        detected_objects = process_obstacles(frame);
        auto ty1 = std::chrono::steady_clock::now();
        s_last_yolo_ms =
            std::chrono::duration<double, std::milli>(ty1 - ty0).count();
    }


    adas::core::WorldModel world_model;

    world_model.timestamp_us = adas::core::get_current_timestamp_us();

    world_model.frame_seq = frame_counter++;

    world_model.current_lane = lane.value_or(adas::core::Lane());

    world_model.obstacles = detected_objects;

    return world_model;
}

}  // namespace adas::perception