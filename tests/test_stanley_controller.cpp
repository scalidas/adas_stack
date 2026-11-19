#include "controller/stanley_controller.hpp"
#include "perception/lane_detection.hpp"
#include "planning/planner.hpp"
#include "streaming/websocket_streamer.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

int main() {
    std::cout << "[Test] Starting Stanley Controller test suite..." << std::endl;

    // 1. Test default construction and reasonable default values
    {
        adas::controller::StanleyController controller;
        assert(std::abs(controller.get_stanley_constant() - 1.0) < 1e-6);
        assert(std::abs(controller.stanley_constant() - 1.0) < 1e-6);
        assert(std::abs(controller.get_k() - 1.0) < 1e-6);
        assert(controller.get_max_steering_angle() > 0.0);
        assert(controller.get_softening_constant() > 0.0);
        std::cout << "[Test] Default construction passed." << std::endl;
    }

    // 2. Test getter and setter for Stanley constant and parameters
    {
        adas::controller::StanleyController controller;
        controller.set_stanley_constant(2.5);
        assert(std::abs(controller.get_stanley_constant() - 2.5) < 1e-6);
        assert(std::abs(controller.stanley_constant() - 2.5) < 1e-6);

        controller.set_max_steering_angle(0.65);
        assert(std::abs(controller.get_max_steering_angle() - 0.65) < 1e-6);

        controller.set_softening_constant(0.25);
        assert(std::abs(controller.get_softening_constant() - 0.25) < 1e-6);

        controller.set_wheelbase(1.8);
        assert(std::abs(controller.get_wheelbase() - 1.8) < 1e-6);
        std::cout << "[Test] Getters and setters passed." << std::endl;
    }

    // 3. Test constructor from configuration file with stanley constant
    {
        const std::string test_cfg_path = "test_controller_config.json";
        {
            std::ofstream out(test_cfg_path);
            out << "{\n"
                << "  \"stanley_constant\": 1.75,\n"
                << "  \"max_steering_angle\": 0.45,\n"
                << "  \"softening_constant\": 0.2\n"
                << "}\n";
        }

        adas::controller::StanleyController controller(test_cfg_path);
        assert(std::abs(controller.get_stanley_constant() - 1.75) < 1e-6);
        assert(std::abs(controller.get_max_steering_angle() - 0.45) < 1e-6);
        assert(std::abs(controller.get_softening_constant() - 0.2) < 1e-6);

        std::remove(test_cfg_path.c_str());
        std::cout << "[Test] Construction from JSON config file passed." << std::endl;
    }

    // 4. Test fallback to default when config file is missing or lacks the constant
    {
        // Non-existent file
        adas::controller::StanleyController ctrl_missing("non_existent_config.json");
        assert(std::abs(ctrl_missing.get_stanley_constant() - 1.0) < 1e-6);

        // File exists but does not specify stanley constant (like live_lane_demo.json)
        adas::controller::StanleyController ctrl_fallback("config/live_lane_demo.json");
        assert(std::abs(ctrl_fallback.get_stanley_constant() - 1.0) < 1e-6);

        std::cout << "[Test] Fallback to default Stanley constant passed." << std::endl;
    }

    // 5. Test construction from PerceptionConfig struct
    {
        adas::config::PerceptionConfig cfg;
        cfg.stanley_k = 3.2;
        cfg.max_steering_angle = 0.5;
        cfg.softening_constant = 0.15;

        adas::controller::StanleyController controller(cfg);
        assert(std::abs(controller.get_stanley_constant() - 3.2) < 1e-6);
        assert(std::abs(controller.get_max_steering_angle() - 0.5) < 1e-6);
        assert(std::abs(controller.get_softening_constant() - 0.15) < 1e-6);
        std::cout << "[Test] Construction from PerceptionConfig passed." << std::endl;
    }

    // 6. Test Stanley control law: Straight path with 0 error
    {
        adas::controller::StanleyController controller(1.0);

        adas::core::Trajectory traj;
        for (int i = 0; i < 10; ++i) {
            traj.points.push_back({static_cast<double>(i) * 0.5, 0.0, 0.0, 2.0});
        }

        adas::core::VehicleControlCommand cmd = controller.compute_control_command(traj);
        assert(std::abs(cmd.target_steering_angle) < 1e-4);
        assert(std::abs(cmd.target_velocity - 2.0) < 1e-6);
        assert(!cmd.emergency_brake);
        std::cout << "[Test] Zero error straight tracking passed: steering=" << cmd.target_steering_angle << std::endl;
    }

    // 7. Test Stanley control law: Cross-track error
    {
        adas::controller::StanleyController controller(1.0);

        // Path is 1.0 meter to the right (+y)
        adas::core::Trajectory traj_right;
        for (int i = 0; i < 10; ++i) {
            traj_right.points.push_back({static_cast<double>(i) * 0.5, 1.0, 0.0, 2.0});
        }

        // Path is 1.0 meter to the left (-y)
        adas::core::Trajectory traj_left;
        for (int i = 0; i < 10; ++i) {
            traj_left.points.push_back({static_cast<double>(i) * 0.5, -1.0, 0.0, 2.0});
        }

        adas::core::VehicleControlCommand cmd_right = controller.compute_control_command(traj_right);
        adas::core::VehicleControlCommand cmd_left = controller.compute_control_command(traj_left);

        // When path is to the right, vehicle should steer right (+delta)
        assert(cmd_right.target_steering_angle > 0.0);
        // When path is to the left, vehicle should steer left (-delta)
        assert(cmd_left.target_steering_angle < 0.0);
        // Symmetric
        assert(std::abs(cmd_right.target_steering_angle + cmd_left.target_steering_angle) < 1e-6);

        std::cout << "[Test] Cross-track steering passed: right=" << cmd_right.target_steering_angle
                  << ", left=" << cmd_left.target_steering_angle << std::endl;
    }

    // 8. Test Stanley control law: Heading error
    {
        adas::controller::StanleyController controller(1.0);

        // Straight path along x
        adas::core::Trajectory straight_traj;
        for (int i = 0; i < 10; ++i) {
            straight_traj.points.push_back({static_cast<double>(i) * 0.5, 0.0, 0.0, 2.0});
        }

        // Vehicle heading angled +0.1 rad to the right
        adas::core::EgoState ego_right;
        ego_right.x = 0.0;
        ego_right.y = 0.0;
        ego_right.yaw = 0.1;
        ego_right.velocity = 2.0;

        adas::core::VehicleControlCommand cmd = controller.compute_control_command(straight_traj, ego_right);
        // Heading error = 0.0 - 0.1 = -0.1. Cross-track is 0. Steering should be negative (steer left).
        assert(cmd.target_steering_angle < 0.0);
        assert(std::abs(cmd.target_steering_angle - (-0.1)) < 1e-4);

        std::cout << "[Test] Pure heading error tracking passed: steering=" << cmd.target_steering_angle << std::endl;
    }

    // 9. Test steering angle clamping
    {
        adas::controller::StanleyController controller(1.0, 0.4); // max steer 0.4 rad

        adas::core::Trajectory traj_far;
        for (int i = 0; i < 10; ++i) {
            traj_far.points.push_back({static_cast<double>(i) * 0.5, 100.0, 0.0, 2.0});
        }

        adas::core::VehicleControlCommand cmd = controller.compute_control_command(traj_far);
        assert(std::abs(cmd.target_steering_angle - 0.4) < 1e-6);
        std::cout << "[Test] Steering angle clamping passed: angle=" << cmd.target_steering_angle << std::endl;
    }

    // 10. Test emergency stop handling
    {
        adas::controller::StanleyController controller;

        adas::core::Trajectory estop_traj;
        estop_traj.is_emergency_stop = true;
        for (int i = 0; i < 5; ++i) {
            estop_traj.points.push_back({static_cast<double>(i) * 0.5, 1.0, 0.0, 2.0});
        }

        adas::core::VehicleControlCommand cmd = controller.compute_control_command(estop_traj);
        assert(cmd.emergency_brake == true);
        assert(cmd.target_velocity == 0.0);
        assert(cmd.target_steering_angle == 0.0);

        // Empty trajectory
        adas::core::Trajectory empty_traj;
        adas::core::VehicleControlCommand cmd_empty = controller.compute_control_command(empty_traj);
        assert(cmd_empty.target_velocity == 0.0);
        std::cout << "[Test] Emergency stop handling passed." << std::endl;
    }

    // 11. Test Planner to StanleyController pipeline integration
    {
        adas::config::PerceptionConfig config;
        config.num_poses = 10;
        config.path_step_size = 0.5;
        config.cruise_velocity = 2.0;
        config.stanley_k = 1.2;

        adas::planning::Planner planner(config);
        adas::core::WorldModel wm;
        // Lane shifted 0.8m to the right
        wm.current_lane.centerline = {.c0 = 0.8, .c1 = 0.0, .c2 = 0.0};

        adas::core::Trajectory planned_traj = planner.plan_trajectory(wm);
        assert(planned_traj.points.size() == 10);

        adas::controller::StanleyController controller(config);
        adas::core::VehicleControlCommand cmd = controller.compute_control_command(planned_traj);

        assert(cmd.target_velocity == 2.0);
        assert(cmd.target_steering_angle > 0.0); // should steer right toward centerline
        assert(!cmd.emergency_brake);
        std::cout << "[Test] Planner -> Controller integration passed: steering="
                  << cmd.target_steering_angle << ", vel=" << cmd.target_velocity << std::endl;
    }

    // 12. Test VehicleControlCommand JSON serialization and deserialization
    {
        adas::core::VehicleControlCommand cmd;
        cmd.timestamp_us = 987654;
        cmd.target_steering_angle = 0.25;
        cmd.target_velocity = 3.5;
        cmd.target_acceleration = 0.5;
        cmd.emergency_brake = false;

        nlohmann::json j;
        adas::core::to_json(j, cmd);
        assert(j["timestamp_us"] == 987654);
        assert(j["target_steering_angle"] == 0.25);
        assert(j["target_velocity"] == 3.5);
        assert(j["emergency_brake"] == false);

        adas::core::VehicleControlCommand parsed;
        adas::core::from_json(j, parsed);
        assert(parsed.timestamp_us == 987654);
        assert(parsed.target_steering_angle == 0.25);
        assert(parsed.target_velocity == 3.5);
        assert(parsed.emergency_brake == false);

        std::cout << "[Test] VehicleControlCommand serialization passed." << std::endl;
    }

    // 13. Test Stop Sign Planning and Velocity Scaling
    {
        adas::config::PerceptionConfig config;
        config.num_poses = 10;
        config.path_step_size = 0.2;
        config.cruise_velocity = 2.0;
        config.enable_stop_signs = true;
        config.stop_sign_distance = 2.0;
        config.stop_sign_buffer = 0.4;
        config.stop_sign_hold_time_s = 1.0;

        adas::planning::Planner planner(config);
        adas::core::WorldModel wm;

        // A. Without stop signs in obstacles -> cruise velocity
        adas::core::Trajectory traj_no_sign = planner.plan_trajectory(wm);
        assert(traj_no_sign.points[0].target_velocity == 2.0);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::CRUISING);

        // B. With stop sign outside stop_sign_distance (e.g. 3.0m) -> cruise velocity
        adas::core::TrackedObject sign_far;
        sign_far.label = "stop sign";
        sign_far.position.x = 3.0;
        wm.obstacles.push_back(sign_far);

        adas::core::Trajectory traj_far = planner.plan_trajectory(wm);
        assert(traj_far.points[0].target_velocity == 2.0);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::CRUISING);

        // C. With stop sign within deceleration zone (e.g. 1.2m) -> decelerates
        wm.obstacles[0].position.x = 1.2;
        adas::core::Trajectory traj_decel = planner.plan_trajectory(wm);
        // Distance to stop line = 1.2 - 0.4 = 0.8m. Total decel distance = 2.0 - 0.4 = 1.6m.
        // Ratio at pose 0 = 0.8 / 1.6 = 0.5 -> vel = 1.0m/s.
        assert(std::abs(traj_decel.points[0].target_velocity - 1.0) < 1e-4);
        // Waypoint at x=0.8m is at stop line -> velocity = 0.0
        assert(traj_decel.points[4].target_velocity == 0.0);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::CRUISING);

        // D. Reached stop buffer (e.g. 0.42m -> dist to stop = 0.02m <= 0.05m) -> STOPPED
        wm.obstacles[0].position.x = 0.42;
        adas::core::Trajectory traj_stopped = planner.plan_trajectory(wm);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::STOPPED);
        for (const auto& pt : traj_stopped.points) {
            assert(pt.target_velocity == 0.0);
        }

        // Still stopped before 1.0 second expires
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        traj_stopped = planner.plan_trajectory(wm);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::STOPPED);
        assert(traj_stopped.points[0].target_velocity == 0.0);

        // E. After 1.0 second has elapsed -> PROCEEDING
        std::this_thread::sleep_for(std::chrono::milliseconds(850));
        adas::core::Trajectory traj_proceeding = planner.plan_trajectory(wm);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::PROCEEDING);
        assert(traj_proceeding.points[0].target_velocity == 2.0);

        // F. Stop sign leaves the frame -> back to CRUISING
        wm.obstacles.clear();
        adas::core::Trajectory traj_cleared = planner.plan_trajectory(wm);
        assert(planner.get_stop_sign_state() == adas::planning::StopSignState::CRUISING);
        assert(traj_cleared.points[0].target_velocity == 2.0);

        std::cout << "[Test] Stop sign planning, deceleration, 1s pause, and proceed passed." << std::endl;
    }

    // 9. Test ransac_downscale config serialization and lane detection with downscaling
    {
        // A. Config serialization and deserialization
        adas::config::PerceptionConfig default_cfg;
        assert(std::abs(default_cfg.ransac_downscale - 1.0) < 1e-6);

        nlohmann::json j_float = {{"ransac_downscale", 4.0}};
        adas::config::PerceptionConfig cfg_float;
        adas::config::from_json(j_float, cfg_float);
        assert(std::abs(cfg_float.ransac_downscale - 4.0) < 1e-6);

        nlohmann::json j_int = {{"ransac_downscale", 2}};
        adas::config::PerceptionConfig cfg_int;
        adas::config::from_json(j_int, cfg_int);
        assert(std::abs(cfg_int.ransac_downscale - 2.0) < 1e-6);

        nlohmann::json serialized = cfg_float;
        assert(serialized.contains("ransac_downscale"));
        assert(std::abs(serialized["ransac_downscale"].get<double>() - 4.0) < 1e-6);

        // B. Test detect_lanes with synthetic 640x480 BEV image
        cv::Mat bev = cv::Mat::zeros(480, 640, CV_8UC3);
        // Draw two vertical white lane lines at x = 220 and x = 420
        cv::line(bev, cv::Point(220, 100), cv::Point(220, 450), cv::Scalar(255, 255, 255), 6);
        cv::line(bev, cv::Point(420, 100), cv::Point(420, 450), cv::Scalar(255, 255, 255), 6);

        adas::config::PerceptionConfig test_cfg;
        test_cfg.white_pixel_threshold = 200;
        test_cfg.ransac_iterations = 50;
        test_cfg.ransac_inlier_threshold = 15.0f;
        test_cfg.min_inlier_count = 500;

        // Test at full resolution (ransac_downscale = 1.0)
        test_cfg.ransac_downscale = 1.0;
        auto lanes_full = adas::perception::detect_lanes(bev, test_cfg);
        assert(lanes_full.size() == 2);
        double full_x1 = lanes_full[0].evaluate(300.0);
        double full_x2 = lanes_full[1].evaluate(300.0);
        if (full_x1 > full_x2) std::swap(full_x1, full_x2);
        assert(std::abs(full_x1 - 220.0) < 5.0);
        assert(std::abs(full_x2 - 420.0) < 5.0);

        // Test at reduced resolution (ransac_downscale = 4.0)
        test_cfg.ransac_downscale = 4.0;
        auto lanes_downscaled = adas::perception::detect_lanes(bev, test_cfg);
        assert(lanes_downscaled.size() == 2);
        double down_x1 = lanes_downscaled[0].evaluate(300.0);
        double down_x2 = lanes_downscaled[1].evaluate(300.0);
        if (down_x1 > down_x2) std::swap(down_x1, down_x2);
        assert(std::abs(down_x1 - 220.0) < 6.0);
        assert(std::abs(down_x2 - 420.0) < 6.0);

        std::cout << "[Test] ransac_downscale config and reduced-resolution lane detection passed." << std::endl;
    }

    std::cout << "[Test] ALL STANLEY CONTROLLER TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
