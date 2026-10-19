#include "planning/planner.hpp"
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

    //Populate the poses in the trajectory
    for (int i = 0; i < num_poses; i++) {
        double x = i * path_step_size;

        trajectory.points.push_back(adas::core::TrajectoryPoint(
            x, center_line.evaluate(x), 0.0, cruise_velocity));
    }

    return trajectory;
}
}  // namespace adas::planning