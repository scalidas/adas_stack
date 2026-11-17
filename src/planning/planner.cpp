#include "planning/planner.hpp"
#include <algorithm>
#include <cmath>
#include "core/core_types.hpp"

namespace adas::planning {

Planner::Planner(const adas::config::PerceptionConfig& config)
    : num_poses(config.num_poses),
      path_step_size(config.path_step_size),
      cruise_velocity(config.cruise_velocity),
      enable_stop_signs(config.enable_stop_signs),
      stop_sign_distance(config.stop_sign_distance),
      stop_sign_buffer(config.stop_sign_buffer),
      stop_sign_hold_time_s(config.stop_sign_hold_time_s) {}

adas::core::Trajectory Planner::plan_trajectory(
    const adas::core::WorldModel& world_model) {

    adas::core::PolynomialLaneBoundary center_line =
        world_model.current_lane.centerline;

    adas::core::Trajectory trajectory;
    trajectory.timestamp_us = adas::core::get_current_timestamp_us();

    // Find the nearest stop sign
    std::optional<double> nearest_stop_sign_dist;
    if (enable_stop_signs) {
        for (const auto& obj : world_model.obstacles) {
            if (obj.label == "stop sign" && obj.position.x > 0.0) {
                if (!nearest_stop_sign_dist.has_value() ||
                    obj.position.x < nearest_stop_sign_dist.value()) {
                    nearest_stop_sign_dist = obj.position.x;
                }
            }
        }
    }

    // Handle stop sign based on state
    double effective_sign_dist = -1.0;

    if (enable_stop_signs) {
        auto now = std::chrono::steady_clock::now();

        switch (stop_state_) {
            case StopSignState::CRUISING:
                if (nearest_stop_sign_dist.has_value()) {
                    double sign_dist = nearest_stop_sign_dist.value();
                    double dist_to_stop = sign_dist - stop_sign_buffer;
                    if (sign_dist <= stop_sign_distance) {
                        if (dist_to_stop <= 0.05) {
                            // Reached the stopping distance - stop the car
                            stop_state_ = StopSignState::STOPPED;
                            stop_start_time_ = now;
                        } else {
                            effective_sign_dist = sign_dist;
                        }
                    }
                }
                break;

            case StopSignState::STOPPED: {
                double elapsed_s =
                    std::chrono::duration<double>(now - stop_start_time_)
                        .count();
                if (elapsed_s >= stop_sign_hold_time_s) {
                    stop_state_ = nearest_stop_sign_dist.has_value()
                                      ? StopSignState::PROCEEDING
                                      : StopSignState::CRUISING;
                }
                break;
            }

            case StopSignState::PROCEEDING:
                if (!nearest_stop_sign_dist.has_value()) {
                    stop_state_ = StopSignState::CRUISING;
                }
                break;
        }
    }

    // Check if centerline is in BEV pixel coordinates (c0 > 10.0) or already in metric coordinates
    constexpr double METERS_PER_BEV_PIXEL = 0.00084;
    constexpr double BEV_ORIGIN_Y = 400.0;  // Front of vehicle in BEV pixels
    constexpr double BEV_CENTER_X = 320.0;  // Vehicle center in BEV pixels

    bool is_bev_pixels = (std::abs(center_line.c0) > 10.0);

    // Populate the poses in the trajectory
    for (int i = 0; i < num_poses; i++) {
        double x = i * path_step_size;
        double y = 0.0;

        if (is_bev_pixels) {
            double y_bev = BEV_ORIGIN_Y - (x / METERS_PER_BEV_PIXEL);
            double x_bev = center_line.evaluate(y_bev);
            y = (x_bev - BEV_CENTER_X) * METERS_PER_BEV_PIXEL;
        } else {
            y = center_line.evaluate(x);
        }

        // Velocity scaling at stop signs if needed
        double target_velocity = cruise_velocity;
        if (enable_stop_signs) {
            if (stop_state_ == StopSignState::STOPPED) {
                target_velocity = 0.0;
            } else if (stop_state_ == StopSignState::CRUISING &&
                       effective_sign_dist > 0.0) {
                double d_stop_line = effective_sign_dist - stop_sign_buffer;
                double d_rem = d_stop_line - x;
                double d_decel =
                    std::max(0.05, stop_sign_distance - stop_sign_buffer);

                if (d_rem <= 0.0) {
                    target_velocity = 0.0;
                } else {
                    double ratio = std::clamp(d_rem / d_decel, 0.0, 1.0);
                    target_velocity = cruise_velocity * ratio;
                }
            }
        }

        trajectory.points.push_back(
            adas::core::TrajectoryPoint(x, y, 0.0, target_velocity));
    }

    return trajectory;
}
}  // namespace adas::planning