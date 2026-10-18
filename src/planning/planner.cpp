#include "planning/planner.hpp"
#include "core/core_types.hpp"

namespace adas::planning {

    adas::core::Trajectory Planner::plan_trajectory(
    const adas::core::WorldModel& world_model) {
    
    adas::core::PolynomialLaneBoundary center_line = world_model.current_lane.centerline;

    adas::core::Trajectory trajectory;

    //Populate the poses in the trajectory

    return trajectory;
}
}