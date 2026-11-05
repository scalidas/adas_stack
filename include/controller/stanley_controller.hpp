#pragma once

#include <cmath>
#include <string>
#include <nlohmann/json.hpp>
#include "config/config.hpp"
#include "core/core_types.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace adas::controller {

/**
 * @brief Stanley Path Tracking Controller
 * 
 * Implements the Stanley control law formulated by Stanford's DARPA Grand Challenge team.
 * Calculates vehicle steering commands based on cross-track error at the front axle
 * and heading error relative to the trajectory path.
 * 
 * Control law:
 *   delta = heading_error + atan2(k * cross_track_error, velocity + softening_constant)
 * 
 * Coordinate frame:
 *   +x: forward
 *   +y: right
 *   +z: up
 *   Positive steering/yaw turns right (clockwise from top view).
 */
class StanleyController {
public:
    static constexpr double DEFAULT_STANLEY_K = 1.0;
    static constexpr double DEFAULT_MAX_STEERING_ANGLE = 0.5235987755982988;  // ~30 degrees in radians
    static constexpr double DEFAULT_SOFTENING_CONSTANT = 0.1;                 // 0.1 m/s velocity softening
    static constexpr double DEFAULT_WHEELBASE = 0.0;                          // Front axle centered in vehicle frame

    // Default constructor: uses reasonable default parameters
    StanleyController();

    // Construct from JSON configuration file path.
    // Loads stanley constant/coefficient if present, or falls back to reasonable defaults.
    explicit StanleyController(const std::string& config_path);
    explicit StanleyController(const char* config_path);

    // Construct from PerceptionConfig struct
    explicit StanleyController(const adas::config::PerceptionConfig& config);

    // Construct directly from parsed JSON object
    explicit StanleyController(const nlohmann::json& json_config);

    // Construct with explicit parameters
    explicit StanleyController(double stanley_k,
                               double max_steering_angle = DEFAULT_MAX_STEERING_ANGLE,
                               double softening_constant = DEFAULT_SOFTENING_CONSTANT,
                               double wheelbase = DEFAULT_WHEELBASE);

    // Consumes a Trajectory and computes the steering and velocity control command
    adas::core::VehicleControlCommand compute_control_command(
        const adas::core::Trajectory& trajectory);

    // Overload taking explicit vehicle EgoState
    adas::core::VehicleControlCommand compute_control_command(
        const adas::core::Trajectory& trajectory,
        const adas::core::EgoState& ego_state);

    // API aliases for convenience
    adas::core::VehicleControlCommand compute_command(
        const adas::core::Trajectory& trajectory) {
        return compute_control_command(trajectory);
    }
    adas::core::VehicleControlCommand compute_command(
        const adas::core::Trajectory& trajectory,
        const adas::core::EgoState& ego_state) {
        return compute_control_command(trajectory, ego_state);
    }
    adas::core::VehicleControlCommand calculate_control_command(
        const adas::core::Trajectory& trajectory) {
        return compute_control_command(trajectory);
    }
    adas::core::VehicleControlCommand calculate_control_command(
        const adas::core::Trajectory& trajectory,
        const adas::core::EgoState& ego_state) {
        return compute_control_command(trajectory, ego_state);
    }
    adas::core::VehicleControlCommand control(
        const adas::core::Trajectory& trajectory) {
        return compute_control_command(trajectory);
    }
    adas::core::VehicleControlCommand control(
        const adas::core::Trajectory& trajectory,
        const adas::core::EgoState& ego_state) {
        return compute_control_command(trajectory, ego_state);
    }

    // Stanley constant getters and setters
    double get_stanley_constant() const { return stanley_k_; }
    void set_stanley_constant(double k) { stanley_k_ = k; }

    double stanley_constant() const { return stanley_k_; }
    double get_stanley_k() const { return stanley_k_; }
    void set_stanley_k(double k) { stanley_k_ = k; }
    double get_k() const { return stanley_k_; }
    void set_k(double k) { stanley_k_ = k; }

    // Auxiliary parameter getters and setters
    double get_max_steering_angle() const { return max_steering_angle_; }
    void set_max_steering_angle(double max_angle) { max_steering_angle_ = max_angle; }

    double get_softening_constant() const { return softening_constant_; }
    void set_softening_constant(double ks) { softening_constant_ = ks; }

    double get_wheelbase() const { return wheelbase_; }
    void set_wheelbase(double wb) { wheelbase_ = wb; }

    // Load or reload from a JSON file path
    bool load_config_file(const std::string& config_path);

private:
    double stanley_k_{DEFAULT_STANLEY_K};
    double max_steering_angle_{DEFAULT_MAX_STEERING_ANGLE};
    double softening_constant_{DEFAULT_SOFTENING_CONSTANT};
    double wheelbase_{DEFAULT_WHEELBASE};

    void parse_json_config(const nlohmann::json& j);
};

// Convenient type aliases
using Stanley = StanleyController;
using Controller = StanleyController;

}  // namespace adas::controller
