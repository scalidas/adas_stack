#pragma once

#include <chrono>
#include "config/config.hpp"
#include "core/core_types.hpp"

namespace adas::planning {

enum class StopSignState {
    CRUISING,
    STOPPED,
    PROCEEDING
};

class Planner {
   private:
    int num_poses;
    double path_step_size;
    double cruise_velocity;

    bool enable_stop_signs{false};
    double stop_sign_distance{2.0};
    double stop_sign_buffer{0.3};
    double stop_sign_hold_time_s{1.0};

    StopSignState stop_state_{StopSignState::CRUISING};
    std::chrono::steady_clock::time_point stop_start_time_{};

   public:
    explicit Planner(const adas::config::PerceptionConfig& config = {});

    // Plan a feasible trajectory given the current world model
    adas::core::Trajectory plan_trajectory(
        const adas::core::WorldModel& world_model);

    StopSignState get_stop_sign_state() const { return stop_state_; }
    void reset_stop_sign_state() { stop_state_ = StopSignState::CRUISING; }
};

}  // namespace adas::planning
