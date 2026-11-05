#include "controller/stanley_controller.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

namespace adas::controller {

StanleyController::StanleyController()
    : stanley_k_(DEFAULT_STANLEY_K),
      max_steering_angle_(DEFAULT_MAX_STEERING_ANGLE),
      softening_constant_(DEFAULT_SOFTENING_CONSTANT),
      wheelbase_(DEFAULT_WHEELBASE) {}

StanleyController::StanleyController(const std::string& config_path)
    : stanley_k_(DEFAULT_STANLEY_K),
      max_steering_angle_(DEFAULT_MAX_STEERING_ANGLE),
      softening_constant_(DEFAULT_SOFTENING_CONSTANT),
      wheelbase_(DEFAULT_WHEELBASE) {
    if (!config_path.empty()) {
        if (!load_config_file(config_path)) {
            std::cout << "[StanleyController] Could not load or parse config at: "
                      << config_path << ". Falling back to default Stanley constant: "
                      << stanley_k_ << std::endl;
        } else {
            std::cout << "[StanleyController] Loaded Stanley constant ("
                      << stanley_k_ << ") from " << config_path << std::endl;
        }
    }
}

StanleyController::StanleyController(const char* config_path)
    : StanleyController(config_path ? std::string(config_path) : std::string{}) {}

StanleyController::StanleyController(const adas::config::PerceptionConfig& config)
    : stanley_k_(config.stanley_k),
      max_steering_angle_(config.max_steering_angle),
      softening_constant_(config.softening_constant),
      wheelbase_(config.wheelbase) {}

StanleyController::StanleyController(const nlohmann::json& json_config)
    : stanley_k_(DEFAULT_STANLEY_K),
      max_steering_angle_(DEFAULT_MAX_STEERING_ANGLE),
      softening_constant_(DEFAULT_SOFTENING_CONSTANT),
      wheelbase_(DEFAULT_WHEELBASE) {
    parse_json_config(json_config);
}

StanleyController::StanleyController(double stanley_k,
                                     double max_steering_angle,
                                     double softening_constant,
                                     double wheelbase)
    : stanley_k_(stanley_k),
      max_steering_angle_(max_steering_angle),
      softening_constant_(softening_constant),
      wheelbase_(wheelbase) {}

void StanleyController::parse_json_config(const nlohmann::json& j) {
    // Check multiple possible key names for Stanley constant/gain/coefficient
    if (j.contains("stanley_k")) {
        stanley_k_ = j["stanley_k"].get<double>();
    } else if (j.contains("stanley_constant")) {
        stanley_k_ = j["stanley_constant"].get<double>();
    } else if (j.contains("stanley_coefficient")) {
        stanley_k_ = j["stanley_coefficient"].get<double>();
    } else if (j.contains("stanley_gain")) {
        stanley_k_ = j["stanley_gain"].get<double>();
    } else if (j.contains("k")) {
        stanley_k_ = j["k"].get<double>();
    }

    if (j.contains("max_steering_angle")) {
        max_steering_angle_ = j["max_steering_angle"].get<double>();
    }
    if (j.contains("softening_constant")) {
        softening_constant_ = j["softening_constant"].get<double>();
    }
    if (j.contains("wheelbase")) {
        wheelbase_ = j["wheelbase"].get<double>();
    }
}

bool StanleyController::load_config_file(const std::string& config_path) {
    if (config_path.empty()) {
        return false;
    }
    std::ifstream file(config_path);
    if (!file.is_open()) {
        return false;
    }
    try {
        nlohmann::json j;
        file >> j;
        parse_json_config(j);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[StanleyController] Exception while parsing config: "
                  << e.what() << std::endl;
        return false;
    }
}

adas::core::VehicleControlCommand StanleyController::compute_control_command(
    const adas::core::Trajectory& trajectory) {
    // Default ego state: centered at origin in vehicle coordinates
    adas::core::EgoState default_ego;
    default_ego.x = 0.0;
    default_ego.y = 0.0;
    default_ego.yaw = 0.0;
    default_ego.velocity = 0.0;
    default_ego.steering_angle = 0.0;

    return compute_control_command(trajectory, default_ego);
}

adas::core::VehicleControlCommand StanleyController::compute_control_command(
    const adas::core::Trajectory& trajectory,
    const adas::core::EgoState& ego_state) {

    adas::core::VehicleControlCommand cmd;
    cmd.timestamp_us = adas::core::get_current_timestamp_us();

    if (trajectory.points.empty()) {
        cmd.target_steering_angle = 0.0;
        cmd.target_velocity = 0.0;
        cmd.target_acceleration = 0.0;
        cmd.emergency_brake = trajectory.is_emergency_stop;
        return cmd;
    }

    if (trajectory.is_emergency_stop) {
        cmd.target_steering_angle = 0.0;
        cmd.target_velocity = 0.0;
        cmd.target_acceleration = 0.0;
        cmd.emergency_brake = true;
        return cmd;
    }

    // 1. Calculate front axle position
    const double x_f = ego_state.x + wheelbase_ * std::cos(ego_state.yaw);
    const double y_f = ego_state.y + wheelbase_ * std::sin(ego_state.yaw);
    const double ego_yaw = ego_state.yaw;

    const auto& points = trajectory.points;
    const size_t n = points.size();

    // 2. Find closest segment / projected point on the trajectory
    double min_dist_sq = std::numeric_limits<double>::max();
    double proj_x = points[0].x;
    double proj_y = points[0].y;
    double path_yaw = points[0].yaw;
    double target_vel = points[0].target_velocity;

    if (n == 1) {
        proj_x = points[0].x;
        proj_y = points[0].y;
        path_yaw = points[0].yaw;
        target_vel = points[0].target_velocity;
    } else {
        for (size_t i = 0; i < n - 1; ++i) {
            const double ax = points[i].x;
            const double ay = points[i].y;
            const double bx = points[i + 1].x;
            const double by = points[i + 1].y;

            const double vx = bx - ax;
            const double vy = by - ay;
            const double seg_len_sq = vx * vx + vy * vy;

            double t = 0.0;
            if (seg_len_sq > 1e-12) {
                t = ((x_f - ax) * vx + (y_f - ay) * vy) / seg_len_sq;
                t = std::clamp(t, 0.0, 1.0);
            }

            const double px = ax + t * vx;
            const double py = ay + t * vy;
            const double dx = x_f - px;
            const double dy = y_f - py;
            const double dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                proj_x = px;
                proj_y = py;
                if (seg_len_sq > 1e-12) {
                    path_yaw = std::atan2(vy, vx);
                } else {
                    path_yaw = points[i].yaw;
                }
                target_vel = (1.0 - t) * points[i].target_velocity +
                             t * points[i + 1].target_velocity;
            }
        }
    }

    // 3. Compute heading error: normalized to [-pi, pi]
    auto normalize_angle = [](double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    };
    const double heading_error = normalize_angle(path_yaw - ego_yaw);

    // 4. Compute cross-track error:
    // Vector from front axle to path point:
    // dx = proj_x - x_f, dy = proj_y - y_f.
    // In our coordinate system (+x forward, +y right):
    // The right normal of the path is: n_right = (-sin(path_yaw), cos(path_yaw)).
    // e = dy * cos(path_yaw) - dx * sin(path_yaw)
    const double dx = proj_x - x_f;
    const double dy = proj_y - y_f;
    const double cross_track_error = dy * std::cos(path_yaw) - dx * std::sin(path_yaw);

    // 5. Velocity for cross-track control term
    double v = std::abs(ego_state.velocity);
    if (v < 0.01) {
        v = std::abs(target_vel);
    }

    // 6. Stanley control law:
    // delta = heading_error + atan2(k * cross_track_error, velocity + softening_constant)
    const double cross_track_steering =
        std::atan2(stanley_k_ * cross_track_error, v + softening_constant_);
    double delta = heading_error + cross_track_steering;

    // 7. Steering saturation clamping
    delta = std::clamp(delta, -max_steering_angle_, max_steering_angle_);

    cmd.target_steering_angle = delta;
    cmd.target_velocity = target_vel;
    cmd.target_acceleration = 0.0;
    cmd.emergency_brake = false;

    return cmd;
}

}  // namespace adas::controller
