#include "core/core_types.hpp"

namespace adas::planning {

class Planner {
   private:
   int num_poses; //Number of poses in planned path
   double cruise_velocity; //Cruising velocity

   public:
    //Plan a feasible trajectory given the current world model
    adas::core::Trajectory plan_trajectory(
        const adas::core::WorldModel& world_model);
};

}  // namespace adas::planning
