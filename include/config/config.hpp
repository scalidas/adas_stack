#pragma once

#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace adas::config {

struct PerceptionConfig {
  std::string image_path{"assets/calibration/PXL_20260921_201642411.jpg"};
  std::string intrinsics_path{""};
  std::string extrinsics_path{"config/calibration/extrinsic_calibration.xml"};
  int ransac_iterations{150};
  float ransac_inlier_threshold{10.0f};
  int min_inlier_count{20};
  int white_pixel_threshold{180};
  std::string object_label{"vehicle"};
  float object_bbox_x{750.0f};
  float object_bbox_y{420.0f};
  float object_bbox_width{350.0f};
  float object_bbox_height{240.0f};
  double object_real_height{1.5};

  // Deep Learning / YOLO fields (optional unless running DL demo)
  std::string model_path{"models/yolov8n.onnx"};
  std::string engine_type{"opencv"};
  int webcam_id{0};
  std::vector<int> target_class_ids{0, 2, 3, 5, 7};
  std::map<int, double> class_real_heights;

  // Webserver / Streaming fields
  bool enable_webserver{false};
  bool enable_streaming{false};
  uint16_t webserver_port{9002};
  uint16_t streaming_port{9002};
  std::string webserver_host{"0.0.0.0"};
  std::string streaming_host{"0.0.0.0"};
  bool webserver_logging{false};
  bool streaming_logging{false};

  // Trajectory planning fields
  int num_poses{20};
  double path_step_size{0.5};
  double cruise_velocity{1.0};
};

inline void to_json(nlohmann::json &j, const PerceptionConfig &c) {
  j = nlohmann::json{{"image_path", c.image_path},
                     {"intrinsics_path", c.intrinsics_path},
                     {"extrinsics_path", c.extrinsics_path},
                     {"ransac_iterations", c.ransac_iterations},
                     {"ransac_inlier_threshold", c.ransac_inlier_threshold},
                     {"min_inlier_count", c.min_inlier_count},
                     {"white_pixel_threshold", c.white_pixel_threshold},
                     {"object_label", c.object_label},
                     {"object_bbox_x", c.object_bbox_x},
                     {"object_bbox_y", c.object_bbox_y},
                     {"object_bbox_width", c.object_bbox_width},
                     {"object_bbox_height", c.object_bbox_height},
                     {"object_real_height", c.object_real_height},
                     {"model_path", c.model_path},
                     {"engine_type", c.engine_type},
                     {"webcam_id", c.webcam_id},
                     {"target_class_ids", c.target_class_ids},
                     {"enable_webserver", c.enable_webserver},
                     {"webserver_port", c.webserver_port},
                     {"webserver_host", c.webserver_host},
                     {"webserver_logging", c.webserver_logging},
                     {"num_poses", c.num_poses},
                     {"path_step_size", c.path_step_size},
                     {"cruise_velocity", c.cruise_velocity}};
  if (!c.class_real_heights.empty()) {
    nlohmann::json crh = nlohmann::json::object();
    for (const auto &[k, v] : c.class_real_heights) {
      crh[std::to_string(k)] = v;
    }
    j["class_real_heights"] = crh;
  }
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
  c.object_label = j.value("object_label", c.object_label);
  c.object_bbox_x = j.value("object_bbox_x", c.object_bbox_x);
  c.object_bbox_y = j.value("object_bbox_y", c.object_bbox_y);
  c.object_bbox_width = j.value("object_bbox_width", c.object_bbox_width);
  c.object_bbox_height = j.value("object_bbox_height", c.object_bbox_height);
  c.object_real_height = j.value("object_real_height", c.object_real_height);
  c.model_path = j.value("model_path", c.model_path);
  c.engine_type = j.value("engine_type", c.engine_type);
  c.webcam_id = j.value("webcam_id", c.webcam_id);
  c.num_poses = j.value("num_poses", c.num_poses);
  c.path_step_size = j.value("path_step_size", c.path_step_size);
  c.cruise_velocity = j.value("cruise_velocity", c.cruise_velocity);

  if (j.contains("enable_webserver")) {
    c.enable_webserver = j["enable_webserver"].get<bool>();
    c.enable_streaming = c.enable_webserver;
  } else if (j.contains("enable_streaming")) {
    c.enable_streaming = j["enable_streaming"].get<bool>();
    c.enable_webserver = c.enable_streaming;
  } else if (j.contains("streaming_enabled")) {
    c.enable_streaming = j["streaming_enabled"].get<bool>();
    c.enable_webserver = c.enable_streaming;
  }

  if (j.contains("webserver_port")) {
    c.webserver_port = j["webserver_port"].get<uint16_t>();
    c.streaming_port = c.webserver_port;
  } else if (j.contains("streaming_port")) {
    c.streaming_port = j["streaming_port"].get<uint16_t>();
    c.webserver_port = c.streaming_port;
  } else if (j.contains("port")) {
    c.webserver_port = j["port"].get<uint16_t>();
    c.streaming_port = c.webserver_port;
  }

  if (j.contains("webserver_host")) {
    c.webserver_host = j["webserver_host"].get<std::string>();
    c.streaming_host = c.webserver_host;
  } else if (j.contains("streaming_host")) {
    c.streaming_host = j["streaming_host"].get<std::string>();
    c.webserver_host = c.streaming_host;
  } else if (j.contains("host")) {
    c.webserver_host = j["host"].get<std::string>();
    c.streaming_host = c.webserver_host;
  }

  if (j.contains("webserver_logging")) {
    c.webserver_logging = j["webserver_logging"].get<bool>();
    c.streaming_logging = c.webserver_logging;
  } else if (j.contains("streaming_logging")) {
    c.streaming_logging = j["streaming_logging"].get<bool>();
    c.webserver_logging = c.streaming_logging;
  }

  if (j.contains("target_class_ids")) {
    c.target_class_ids = j["target_class_ids"].get<std::vector<int>>();
  } else if (j.contains("target_coco_ids")) {
    c.target_class_ids = j["target_coco_ids"].get<std::vector<int>>();
  }

  if (j.contains("class_real_heights") && j["class_real_heights"].is_object()) {
    c.class_real_heights.clear();
    for (auto &[key, val] : j["class_real_heights"].items()) {
      try {
        int cid = std::stoi(key);
        c.class_real_heights[cid] = val.get<double>();
      } catch (...) {}
    }
  }
}

// Load configuration from JSON file. If file does not exist, creates it with
// defaults.
PerceptionConfig load_config(const std::string &config_path);

// Save configuration to JSON file.
bool save_config(const std::string &config_path,
                 const PerceptionConfig &config);

} // namespace adas::config
