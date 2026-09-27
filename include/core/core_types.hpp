#pragma once
#include <optional>
#include <string>
#include <vector>

/*
All distances are in m
All speed/velocity is in m/s
+x axis is forward
+y axis is right
+z axis is up
*/

namespace adas::core {

// 1. Perception & World Model Structures

struct Vector3D {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct TrackedObject {
  int id{-1};
  std::string label; // Type of object (vehicle, pedestrian...)
  Vector3D position; // position of the center - in the frame of the vehicle
  Vector3D velocity; // velocity of the object relative to vehicle
  double length{0.0};
  double width{0.0};
  double height{0.0};
  float confidence{0.0f};
};

struct TrafficSign {
  enum class Type { STOP_SIGN, UNKNOWN };
  Type type{Type::UNKNOWN};
  Vector3D position; // position of the center - in the frame of the vehicle
  float confidence{0.0f};
};

// Represent lanes as quadratics in local frame
struct PolynomialLaneBoundary {
  double c0{0.0};
  double c1{0.0};
  double c2{0.0};

  // Evaluate value of lane boundary at a given value
  double evaluate(double x) const { return c0 + (c1 * x) + (c2 * x * x); }
};

struct Lane {
  int lane_id{0};
  PolynomialLaneBoundary left_boundary;
  PolynomialLaneBoundary right_boundary;
  PolynomialLaneBoundary centerline;
  double lane_width{0.50};
  
  // For now, based purely on whether it is a dotted line or not
  bool can_change_left{true};
  bool can_change_right{true};
};

struct EgoState {
  // X and Y are positions in world frame. Z is assumed to always be 0
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double velocity{0.0};       // current speed in +x direction
  double steering_angle{0.0}; // current steering angle
};

struct WorldModel {
  uint64_t timestamp_us{0};
  uint32_t frame_seq{0};
  EgoState ego_state;
  Lane current_lane;
  std::optional<Lane> left_neighbor_lane;
  std::optional<Lane> right_neighbor_lane;
  std::vector<TrackedObject> obstacles;
  std::vector<TrafficSign> traffic_signs;
};

// 2. Planning Structures

struct TrajectoryPoint {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double target_velocity{0.0};
};

struct Trajectory {
  uint64_t timestamp_us{0};
  std::vector<TrajectoryPoint> points;
  bool is_emergency_stop{false};
};

// 3. Control & Actuation Structures

struct VehicleControlCommand {
  uint64_t timestamp_us{0};
  double target_steering_angle{0.0};
  double target_velocity{0.0};
  double target_acceleration{0.0};
  bool emergency_brake{false};
};

} // namespace adas::core