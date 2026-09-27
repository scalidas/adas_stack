#include "config/config.hpp"
#include <fstream>
#include <iostream>

namespace adas::config {

bool save_config(const std::string &config_path, const PerceptionConfig &config) {
  std::ofstream file(config_path);
  if (!file.is_open()) {
    std::cerr << "[Config] Failed to open file for writing: " << config_path << std::endl;
    return false;
  }
  nlohmann::json j = config;
  file << j.dump(4);
  return true;
}

PerceptionConfig load_config(const std::string &config_path) {
  PerceptionConfig config;
  std::ifstream file(config_path);

  if (!file.is_open()) {
    std::cout << "[Config] Config file not found at '" << config_path
              << "'. Creating default configuration file." << std::endl;
    save_config(config_path, config);
    return config;
  }

  try {
    nlohmann::json j;
    file >> j;
    config = j.get<PerceptionConfig>();
    std::cout << "[Config] Successfully loaded configuration from: " << config_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "[Config] Error parsing JSON config file '" << config_path
              << "': " << e.what() << ". Falling back to defaults." << std::endl;
  }

  return config;
}

} // namespace adas::config
