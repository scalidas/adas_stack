#include "planning/planner.hpp"
#include <cmath>
#include "core/core_types.hpp"

namespace adas::planning {

Planner::Planner(const adas::config::PerceptionConfig& config)
    : num_poses(config.num_poses),
      path_step_size(config.path_step_size),
      cruise_velocity(config.cruise_velocity) {}

adas::core::Trajectory Planner::plan_trajectory(
    const adas::core::WorldModel& world_model) {

    adas::core::PolynomialLaneBoundary center_line =
        world_model.current_lane.centerline;

    adas::core::Trajectory trajectory;

    trajectory.timestamp_us = adas::core::get_current_timestamp_us();

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
            // Forward distance x maps to BEV row index y_bev (decreasing row = further forward)
            double y_bev = BEV_ORIGIN_Y - (x / METERS_PER_BEV_PIXEL);
            double x_bev = center_line.evaluate(y_bev);
            y = (x_bev - BEV_CENTER_X) * METERS_PER_BEV_PIXEL;
        } else {
            // Already in metric vehicle coordinates
            y = center_line.evaluate(x);
        }

        trajectory.points.push_back(adas::core::TrajectoryPoint(
            x, y, 0.0, cruise_velocity));
    }

    return trajectory;
}
}  // namespace adas::planning