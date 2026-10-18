#include <chrono>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "calibration/extrinsic_calibration.hpp"
#include "config/config.hpp"
#include "core/core_types.hpp"
#include "perception/lane_detection.hpp"
#include "perception/perceptionengine.hpp"
#include "streaming/websocket_streamer.hpp"
#include "yolo/inference_engine.hpp"
#include "yolo/opencv_dnn_engine.hpp"

static bool is_left(const adas::core::PolynomialLaneBoundary& a,
                    const adas::core::PolynomialLaneBoundary& b) {
    //+y is right
    return a.c0 < b.c0;
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
    stop_webserver();
}

PerceptionEngine::PerceptionEngine(const adas::config::PerceptionConfig& cfg) {
    config = cfg;

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

    if (config.enable_webserver) {
        start_webserver();
    }

    frame_counter = 0;
}

bool PerceptionEngine::start_webserver() {
    if (streamer && streamer->is_running()) {
        return true;
    }
    adas::streaming::StreamerConfig streamer_cfg;
    streamer_cfg.port = config.webserver_port;
    streamer_cfg.host = config.webserver_host;
    streamer_cfg.enable_logging = config.webserver_logging;
    streamer =
        std::make_unique<adas::streaming::WebSocketStreamer>(streamer_cfg);
    if (!streamer->start()) {
        std::cerr << "[PerceptionEngine] Warning: Failed to start webserver on "
                  << config.webserver_host << ":" << config.webserver_port
                  << std::endl;
        return false;
    }
    std::cout << "[PerceptionEngine] Webserver streaming running on ws://"
              << config.webserver_host << ":" << config.webserver_port
              << std::endl;
    return true;
}

void PerceptionEngine::stop_webserver() {
    if (streamer && streamer->is_running()) {
        streamer->stop();
    }
}

bool PerceptionEngine::is_webserver_running() const {
    return streamer && streamer->is_running();
}

void PerceptionEngine::broadcast_world_model(
    const adas::core::WorldModel& world_model) {
    if (streamer && streamer->is_running()) {
        streamer->publish(world_model);
    }
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

    //Construct lane model
    adas::core::Lane lane;
    lane.lane_id = 1;

    //Left boundary is the one smaller at x = 0
    lane.left_boundary = is_left(lane_boundaries[0], lane_boundaries[1])
                             ? lane_boundaries[0]
                             : lane_boundaries[1];
    lane.right_boundary = is_left(lane_boundaries[0], lane_boundaries[1])
                              ? lane_boundaries[1]
                              : lane_boundaries[0];

    constexpr double METERS_PER_BEV_PIXEL = 0.00084;
    lane.left_boundary.c0 *= METERS_PER_BEV_PIXEL;
    lane.right_boundary.c0 *= METERS_PER_BEV_PIXEL;

    double average_lane_pos =
        (lane.left_boundary.c0 + lane.right_boundary.c0) / 2;
    lane.left_boundary.c0 -= average_lane_pos;
    lane.right_boundary.c0 -= average_lane_pos;

    lane.centerline =
        computeCenterline(lane.left_boundary, lane.right_boundary);

    lane.lane_width =
        (lane.right_boundary.evaluate(0) - lane.left_boundary.evaluate(0));

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

adas::core::WorldModel PerceptionEngine::process_frame(const cv::Mat& frame) {
    std::optional<adas::core::Lane> lane;
    std::vector<adas::core::TrackedObject> detected_objects;

    {
        // worker writes to its own dedicated stack variable
        std::jthread worker(
            [&lane, &frame, this] { lane = process_lanes(frame); });

        // Main thread computes the other field locally
        detected_objects = process_obstacles(frame);

        // worker joins automatically here at the closing brace
    }

    adas::core::WorldModel world_model;

    world_model.timestamp_us = adas::core::get_current_timestamp_us();

    world_model.frame_seq = frame_counter++;

    world_model.current_lane = lane.value_or(adas::core::Lane());

    world_model.obstacles = detected_objects;

    return world_model;
}

bool PerceptionEngine::runLiveDemo() {
    if (!inference_engine) {
        inference_engine =
            std::make_unique<adas::yolo::OpenCVDNNEngine>(config.model_path);
    }

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

    if (!config.target_class_ids.empty()) {
        inference_engine->setClasses(config.target_class_ids);
    }

    if (config.enable_webserver && !is_webserver_running()) {
        start_webserver();
    }

    int webcam_id = config.webcam_id;
    if (config.image_path.rfind("webcam", 0) == 0) {
        try {
            std::string idx_str = config.image_path.substr(6);
            size_t first = idx_str.find_first_not_of(" \t");
            if (first != std::string::npos) {
                webcam_id = std::stoi(idx_str.substr(first));
            }
        } catch (...) {
            // keep config.webcam_id
        }
    }

    std::cout << "[PerceptionEngine] Starting live demo on webcam ID: "
              << webcam_id << std::endl;
    cv::VideoCapture cap(webcam_id);
    if (!cap.isOpened()) {
        std::cerr << "[PerceptionEngine] Error: Could not open webcam with ID "
                  << webcam_id << std::endl;
        return false;
    }

    const std::string window_name = "ADAS Perception Engine Live Demo";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat raw_frame;
    cv::Mat inv_ipm = ipm.inv();

    std::cout
        << "[PerceptionEngine] Live demo running. Press 'q' or 'ESC' to exit."
        << std::endl;

    while (true) {
        cap >> raw_frame;
        if (raw_frame.empty()) {
            std::cerr << "[PerceptionEngine] Warning: Captured empty frame."
                      << std::endl;
            break;
        }

        // Standardize input frame to 640x480 matching the homography IPM model
        cv::Mat frame;
        if (raw_frame.cols != 640 || raw_frame.rows != 480) {
            cv::resize(raw_frame, frame, cv::Size(640, 480));
        } else {
            frame = raw_frame;
        }

        // Process frame through PerceptionEngine
        adas::core::WorldModel world_model = process_frame(frame);

        // Serialize and broadcast worldmodel over network through webserver
        if (is_webserver_running()) {
            broadcast_world_model(world_model);
        }

        // Prepare visualization canvas
        cv::Mat display_frame = frame.clone();

        // Project BEV polynomial boundaries back to camera view
        auto project_boundary_to_cam =
            [&](const adas::core::PolynomialLaneBoundary& boundary) {
                std::vector<cv::Point2f> pts_bev;
                for (int y = 140; y <= 420; y += 4) {
                    double x = boundary.evaluate(y);
                    if (x >= 0.0 && x < 640.0) {
                        pts_bev.emplace_back(static_cast<float>(x),
                                             static_cast<float>(y));
                    }
                }
                std::vector<cv::Point> pts_cam_int;
                if (pts_bev.size() >= 2) {
                    std::vector<cv::Point2f> pts_cam;
                    cv::perspectiveTransform(pts_bev, pts_cam, inv_ipm);
                    for (const auto& pt : pts_cam) {
                        if (pt.x >= -50.0f &&
                            pt.x <= display_frame.cols + 50.0f &&
                            pt.y >= 0.0f && pt.y < display_frame.rows) {
                            pts_cam_int.emplace_back(
                                static_cast<int>(std::round(pt.x)),
                                static_cast<int>(std::round(pt.y)));
                        }
                    }
                }
                return pts_cam_int;
            };

        bool has_valid_lane =
            (world_model.current_lane.lane_id > 0) ||
            (world_model.current_lane.left_boundary.c0 != 0.0 ||
             world_model.current_lane.right_boundary.c0 != 0.0);

        if (has_valid_lane) {
            auto left_pts =
                project_boundary_to_cam(world_model.current_lane.left_boundary);
            auto right_pts = project_boundary_to_cam(
                world_model.current_lane.right_boundary);
            auto center_pts =
                project_boundary_to_cam(world_model.current_lane.centerline);

            // Shaded corridor overlay between left and right boundaries
            if (left_pts.size() >= 2 && right_pts.size() >= 2) {
                std::vector<cv::Point> corridor;
                corridor.insert(corridor.end(), left_pts.begin(),
                                left_pts.end());
                corridor.insert(corridor.end(), right_pts.rbegin(),
                                right_pts.rend());

                cv::Mat overlay = display_frame.clone();
                std::vector<std::vector<cv::Point>> polys = {corridor};
                cv::fillPoly(
                    overlay, polys,
                    cv::Scalar(0, 200, 100));  // semi-transparent cyan/green
                cv::addWeighted(overlay, 0.3, display_frame, 0.7, 0,
                                display_frame);
            }

            // Draw left boundary in bright yellow
            if (left_pts.size() >= 2) {
                const cv::Point* pts = left_pts.data();
                int npts = static_cast<int>(left_pts.size());
                cv::polylines(display_frame, &pts, &npts, 1, false,
                              cv::Scalar(0, 215, 255), 3, cv::LINE_AA);
            }

            // Draw right boundary in bright cyan
            if (right_pts.size() >= 2) {
                const cv::Point* pts = right_pts.data();
                int npts = static_cast<int>(right_pts.size());
                cv::polylines(display_frame, &pts, &npts, 1, false,
                              cv::Scalar(255, 200, 0), 3, cv::LINE_AA);
            }

            // Draw centerline in green
            if (center_pts.size() >= 2) {
                const cv::Point* pts = center_pts.data();
                int npts = static_cast<int>(center_pts.size());
                cv::polylines(display_frame, &pts, &npts, 1, false,
                              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }

        // Draw returned obstacles
        for (const auto& obj : world_model.obstacles) {
            cv::Rect bbox_int(static_cast<int>(obj.bounding_box.x),
                              static_cast<int>(obj.bounding_box.y),
                              static_cast<int>(obj.bounding_box.width),
                              static_cast<int>(obj.bounding_box.height));

            if (bbox_int.width > 0 && bbox_int.height > 0) {
                cv::rectangle(display_frame, bbox_int, cv::Scalar(0, 255, 0),
                              2);

                std::string line1 =
                    cv::format("%s %.2f | Fwd: %.2fm", obj.label.c_str(),
                               obj.confidence, obj.position.x);
                std::string line2 =
                    cv::format("Ground: [X=%.2f, Y=%.2f, Z=%.2f]m",
                               obj.position.x, obj.position.y, obj.position.z);

                int font = cv::FONT_HERSHEY_SIMPLEX;
                double scale = 0.45;
                int thick = 1;
                int b1 = 0, b2 = 0;
                cv::Size s1 = cv::getTextSize(line1, font, scale, thick, &b1);
                cv::Size s2 = cv::getTextSize(line2, font, scale, thick, &b2);
                int banner_w = std::max(s1.width, s2.width) + 8;
                int banner_h = s1.height + s2.height + 8;

                int banner_x = std::clamp(
                    bbox_int.x, 0, std::max(0, display_frame.cols - banner_w));
                int banner_y = std::max(banner_h + 2, bbox_int.y);

                cv::rectangle(display_frame,
                              cv::Point(banner_x, banner_y - banner_h),
                              cv::Point(banner_x + banner_w, banner_y),
                              cv::Scalar(25, 25, 25), cv::FILLED);

                cv::putText(display_frame, line1,
                            cv::Point(banner_x + 4, banner_y - s2.height - 4),
                            font, scale, cv::Scalar(0, 255, 0), thick);

                cv::putText(display_frame, line2,
                            cv::Point(banner_x + 4, banner_y - 2), font, scale,
                            cv::Scalar(0, 255, 255), thick);
            }
        }

        // Top HUD bar
        cv::rectangle(display_frame, cv::Point(0, 0),
                      cv::Point(display_frame.cols, 28), cv::Scalar(25, 25, 25),
                      cv::FILLED);
        std::string hud_lane =
            has_valid_lane ? cv::format("Lane: Tracked (W=%.2fm)",
                                        world_model.current_lane.lane_width)
                           : "Lane: None";
        std::string ws_hud = "";
        if (is_webserver_running()) {
            ws_hud =
                cv::format(" | WS: Port %u (%zu clients)", streamer->get_port(),
                           streamer->get_active_client_count());
        }
        std::string hud_info =
            cv::format("Seq: %u | %s | Obstacles: %zu%s | 'q'/ESC to exit",
                       world_model.frame_seq, hud_lane.c_str(),
                       world_model.obstacles.size(), ws_hud.c_str());
        cv::putText(display_frame, hud_info, cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 255), 1,
                    cv::LINE_AA);

        cv::imshow(window_name, display_frame);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            std::cout << "[PerceptionEngine] Live demo stopped by user."
                      << std::endl;
            break;
        }
    }

    if (is_webserver_running()) {
        stop_webserver();
    }

    cap.release();
    cv::destroyWindow(window_name);
    return true;
}

}  // namespace adas::perception