#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace adas::config {

struct PerceptionConfig {
  std::string image_path{"assets/calibration/PXL_20260921_201642411.jpg"};
  std::string intrinsics_path{""};
  std::string extrinsics_path{"config/calibration/extrinsic_calibration.xml"};
  int ransac_iterations{150};
  float ransac_inlier_threshold{10.0f};
  int min_inlier_count{20};
  int white_pixel_threshold{180};
};

inline void to_json(nlohmann::json &j, const PerceptionConfig &c) {
  j = nlohmann::json{
      {"image_path", c.image_path},
      {"intrinsics_path", c.intrinsics_path},
      {"extrinsics_path", c.extrinsics_path},
      {"ransac_iterations", c.ransac_iterations},
      {"ransac_inlier_threshold", c.ransac_inlier_threshold},
      {"min_inlier_count", c.min_inlier_count},
      {"white_pixel_threshold", c.white_pixel_threshold}};
}

inline void from_json(const nlohmann::json &j, PerceptionConfig &c) {
  c.image_path = j.value("image_path", c.image_path);
  c.intrinsics_path = j.value("intrinsics_path", c.intrinsics_path);
  c.extrinsics_path = j.value("extrinsics_path", c.extrinsics_path);
  c.ransac_iterations = j.value("ransac_iterations", c.ransac_iterations);
  c.ransac_inlier_threshold =
      j.value("ransac_inlier_threshold", c.ransac_inlier_threshold);
  c.min_inlier_count = j.value("min_inlier_count", c.min_inlier_count);
  c.white_pixel_threshold =
      j.value("white_pixel_threshold", c.white_pixel_threshold);
}

// Load configuration from JSON file. If file does not exist, creates it with
// defaults.
PerceptionConfig load_config(const std::string &config_path);

// Save configuration to JSON file.
bool save_config(const std::string &config_path,
                 const PerceptionConfig &config);

} // namespace adas::config
