#include "controller/stanley_controller.hpp"
#include "planning/planner.hpp"
#include "streaming/websocket_streamer.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

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

    std::cout << "[Test] ALL STANLEY CONTROLLER TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
