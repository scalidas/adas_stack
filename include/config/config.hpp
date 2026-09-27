#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace adas::config {

struct PerceptionConfig {
  std::string image_path{"assets/calibration/PXL_20260921_201642411.jpg"};
  int ransac_iterations{150};
  float ransac_inlier_threshold{10.0f};
  int min_inlier_count{20};
  int white_pixel_threshold{180};
};

// JSON serialization macros for nlohmann_json
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PerceptionConfig, image_path,
                                   ransac_iterations, ransac_inlier_threshold,
                                   min_inlier_count, white_pixel_threshold)

// Load configuration from JSON file. If file does not exist, creates it with
// defaults.
PerceptionConfig load_config(const std::string &config_path);

// Save configuration to JSON file.
bool save_config(const std::string &config_path,
                 const PerceptionConfig &config);

} // namespace adas::config
