#include "core/orchestrator.hpp"
#include "config/config.hpp"
#include "streaming/websocket_streamer.hpp"

bool adas::core::Orchestrator::start_webserver() {
    if (streamer && streamer->is_running()) {
        return true;
    }
    if (!streamer) {
        return false;
    }
    if (!streamer->start()) {
        std::cerr << "[adas::core::Orchestrator] Warning: Failed to start "
                     "webserver on "
                  << config.webserver_host << ":" << config.webserver_port
                  << std::endl;
        return false;
    }
    std::cout
        << "[adas::core::Orchestrator] Webserver streaming running on ws://"
        << config.webserver_host << ":" << config.webserver_port << std::endl;
    return true;
}

void adas::core::Orchestrator::stop_webserver() {
    if (streamer && streamer->is_running()) {
        streamer->stop();
    }
}

bool adas::core::Orchestrator::is_webserver_running() const {
    return streamer && streamer->is_running();
}

void adas::core::Orchestrator::broadcast_world_model(
    const adas::core::WorldModel& world_model) {
    if (streamer && streamer->is_running()) {
        streamer->publish(world_model);
    }
}

int adas::core::Orchestrator::run() {
    //Spin up webserver
    if (enable_webserver && !is_webserver_running()) {
        start_webserver();
    }

    std::cout << "[Orchestrator] Starting live demo on webcam ID: " << webcam_id
              << std::endl;
    cv::VideoCapture cap(webcam_id);
    if (!cap.isOpened()) {
        std::cerr << "[Orchestrator] Error: Could not open webcam with ID "
                  << webcam_id << std::endl;
        return -1;
    }

    const std::string window_name = "ADAS Perception Engine Live Demo";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat raw_frame;
    cv::Mat inv_ipm = perception_engine.get_ipm().inv();

    std::cout << "[Orchestrator] Live demo running. Press 'q' or 'ESC' to exit."
              << std::endl;

    while (true) {
        cap >> raw_frame;
        if (raw_frame.empty()) {
            std::cerr << "[Orchestrator] Warning: Captured empty frame."
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

        // Process frame through adas::core::Orchestrator
        adas::core::WorldModel world_model =
            perception_engine.process_frame(frame);

        // Plan trajectory through Planner
        world_model.trajectory = planner.plan_trajectory(world_model);

        // Compute vehicle control command using Stanley controller
        adas::core::VehicleControlCommand control_cmd =
            controller.compute_control_command(world_model.trajectory,
                                               world_model.ego_state);
        world_model.control_command = control_cmd;

        //TODO: Move blocking webserver broadcast out of control loop
        // Serialize and broadcast worldmodel (with embedded control command) over network through webserver
        if (is_webserver_running()) {
            broadcast_world_model(world_model);
        }

// Transmit to MCU via SPI
#ifdef ADAS_ENABLE_MCU_LINK
        bool result = mcu_link.send(control_cmd);

//TODO: Handle SPI result failure if needed
#endif

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

        // Bottom HUD bar (Stanley Controller)
        cv::rectangle(display_frame, cv::Point(0, display_frame.rows - 28),
                      cv::Point(display_frame.cols, display_frame.rows),
                      cv::Scalar(25, 25, 25), cv::FILLED);
        std::string hud_ctrl = cv::format(
            "Stanley: Steer=%.2f deg (%.3f rad) | Target V=%.2f m/s | Brake=%s",
            control_cmd.target_steering_angle * 180.0 / 3.14159265358979323846,
            control_cmd.target_steering_angle, control_cmd.target_velocity,
            control_cmd.emergency_brake ? "YES" : "NO");
        cv::putText(display_frame, hud_ctrl,
                    cv::Point(10, display_frame.rows - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 100), 1,
                    cv::LINE_AA);

        cv::imshow(window_name, display_frame);

        int key = cv::waitKey(1);

        if (key == 27 || key == 'q' || key == 'Q') {
            std::cout << "[adas::core::Orchestrator] Live demo stopped by user."
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