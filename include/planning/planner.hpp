#include "config/config.hpp"
#include "core/core_types.hpp"

namespace adas::planning {

class Planner {
   private:
    int num_poses;
    double path_step_size;
    double cruise_velocity;

   public:
    explicit Planner(const adas::config::PerceptionConfig& config = {});

    //Plan a feasible trajectory given the current world model
    adas::core::Trajectory plan_trajectory(
        const adas::core::WorldModel& world_model);
};

}  // namespace adas::planning
