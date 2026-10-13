#include "yolo/opencv_dnn_engine.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "config/config.hpp"
#include "perception/road_object.hpp"

namespace adas::yolo {

namespace {
// Standard 80 COCO class names for YOLO models
const std::vector<std::string> &getDefaultCOCOClasses() {
  static const std::vector<std::string> coco_classes = {
      "person",        "bicycle",       "car",           "motorcycle",
      "airplane",      "bus",           "train",         "truck",
      "boat",          "traffic light", "fire hydrant",  "stop sign",
      "parking meter", "bench",         "bird",          "cat",
      "dog",           "horse",         "sheep",         "cow",
      "elephant",      "bear",          "zebra",         "giraffe",
      "backpack",      "umbrella",      "handbag",       "tie",
      "suitcase",      "frisbee",       "skis",          "snowboard",
      "sports ball",   "kite",          "baseball bat",  "baseball glove",
      "skateboard",    "surfboard",     "tennis racket", "bottle",
      "wine glass",    "cup",           "fork",          "knife",
      "spoon",         "bowl",          "banana",        "apple",
      "sandwich",      "orange",        "broccoli",      "carrot",
      "hot dog",       "pizza",         "donut",         "cake",
      "chair",         "couch",         "potted plant",  "bed",
      "dining table",  "toilet",        "tv",            "laptop",
      "mouse",         "remote",        "keyboard",      "cell phone",
      "microwave",     "oven",          "toaster",       "sink",
      "refrigerator",  "book",          "clock",         "vase",
      "scissors",      "teddy bear",    "hair drier",    "toothbrush"};
  return coco_classes;
}
} // namespace

OpenCVDNNEngine::OpenCVDNNEngine(const std::string &model_path,
                                 float conf_threshold, float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
  loadModel(model_path);
}

bool OpenCVDNNEngine::loadModel(const std::string &model_path) {
  is_loaded_ = false;
  if (model_path.empty()) {
    std::cerr << "[OpenCVDNNEngine] Model path is empty." << std::endl;
    return false;
  }

  try {
    net_ = cv::dnn::readNet(model_path);
    if (net_.empty()) {
      std::cerr << "[OpenCVDNNEngine] Failed to load model from: "
                << model_path << std::endl;
      return false;
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    is_loaded_ = true;
    std::cout << "[OpenCVDNNEngine] Successfully loaded model: " << model_path
              << std::endl;
    return true;
  } catch (const cv::Exception &e) {
    std::cerr << "[OpenCVDNNEngine] OpenCV Exception loading model: "
              << e.what() << std::endl;
    return false;
  } catch (const std::exception &e) {
    std::cerr << "[OpenCVDNNEngine] Exception loading model: " << e.what()
              << std::endl;
    return false;
  }
}

void OpenCVDNNEngine::setConfidenceThreshold(float conf_threshold) {
  conf_threshold_ = conf_threshold;
}

void OpenCVDNNEngine::setNMSThreshold(float nms_threshold) {
  nms_threshold_ = nms_threshold;
}

void OpenCVDNNEngine::setInputSize(int width, int height) {
  input_width_ = width;
  input_height_ = height;
}

void OpenCVDNNEngine::setClasses(const std::vector<std::string> &classes) {
  classes_ = classes;
}

bool OpenCVDNNEngine::isLoaded() const { return is_loaded_; }

float OpenCVDNNEngine::getConfidenceThreshold() const {
  return conf_threshold_;
}

float OpenCVDNNEngine::getNMSThreshold() const { return nms_threshold_; }

std::vector<adas::core::DetectedObject>
OpenCVDNNEngine::infer(const cv::Mat &frame) {
  std::vector<adas::core::DetectedObject> detections;
  if (!is_loaded_ || net_.empty() || frame.empty()) {
    return detections;
  }

  try {
    // Construct 4D blob from input frame
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                           cv::Size(input_width_, input_height_),
                           cv::Scalar(0, 0, 0), true, false);

    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
      return detections;
    }

    const auto &active_classes =
        classes_.empty() ? getDefaultCOCOClasses() : classes_;

    std::vector<cv::Rect2d> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float x_scale = static_cast<float>(frame.cols) / input_width_;
    float y_scale = static_cast<float>(frame.rows) / input_height_;

    for (auto &output_mat : outputs) {
      cv::Mat mat = output_mat;
      if (mat.dims == 3) {
        // e.g. [1, 84, 8400] -> reshape to 2D [84, 8400]
        int sizes[2] = {mat.size[1], mat.size[2]};
        mat = cv::Mat(2, sizes, CV_32F, mat.ptr<float>());
      }

      int rows = mat.rows;
      int cols = mat.cols;

      if (rows == 0 || cols == 0) {
        continue;
      }

      // Check tensor orientation:
      // Orientation A: rows is channels (e.g. 84 or 85), cols is anchors (e.g. 8400)
      if (rows < cols && rows >= 5) {
        int num_anchors = cols;
        int num_channels = rows;

        for (int i = 0; i < num_anchors; ++i) {
          float cx = mat.at<float>(0, i);
          float cy = mat.at<float>(1, i);
          float w = mat.at<float>(2, i);
          float h = mat.at<float>(3, i);

          float max_score = 0.0f;
          int best_class_id = -1;

          // YOLOv8 format: indices 4..num_channels-1 are class scores
          for (int c = 4; c < num_channels; ++c) {
            float score = mat.at<float>(c, i);
            if (score > max_score) {
              max_score = score;
              best_class_id = c - 4;
            }
          }

          if (max_score >= conf_threshold_) {
            float left = (cx - 0.5f * w) * x_scale;
            float top = (cy - 0.5f * h) * y_scale;
            float width = w * x_scale;
            float height = h * y_scale;

            bboxes.emplace_back(left, top, width, height);
            confidences.push_back(max_score);
            class_ids.push_back(best_class_id);
          }
        }
      } else if (rows >= cols && cols >= 5) {
        // Orientation B: rows is anchors, cols is attributes [x, y, w, h, (obj_conf,) class_scores...]
        int num_anchors = rows;
        int num_attrs = cols;

        for (int i = 0; i < num_anchors; ++i) {
          float cx = mat.at<float>(i, 0);
          float cy = mat.at<float>(i, 1);
          float w = mat.at<float>(i, 2);
          float h = mat.at<float>(i, 3);

          float confidence = 0.0f;
          int best_class_id = -1;

          if (num_attrs == 6) {
            // [x, y, w, h, score, class_id]
            confidence = mat.at<float>(i, 4);
            best_class_id = static_cast<int>(mat.at<float>(i, 5));
          } else if (num_attrs > 6) {
            // YOLOv5 format: [x, y, w, h, obj_conf, class0, class1, ...]
            float obj_conf = mat.at<float>(i, 4);
            float max_cls_score = 0.0f;
            for (int c = 5; c < num_attrs; ++c) {
              float score = mat.at<float>(c, i);
              if (score > max_cls_score) {
                max_cls_score = score;
                best_class_id = c - 5;
              }
            }
            confidence = obj_conf * max_cls_score;
          } else if (num_attrs == 5) {
            confidence = mat.at<float>(i, 4);
            best_class_id = 0;
          }

          if (confidence >= conf_threshold_) {
            float left = (cx - 0.5f * w) * x_scale;
            float top = (cy - 0.5f * h) * y_scale;
            float width = w * x_scale;
            float height = h * y_scale;

            bboxes.emplace_back(left, top, width, height);
            confidences.push_back(confidence);
            class_ids.push_back(best_class_id);
          }
        }
      }
    }

    if (bboxes.empty()) {
      return detections;
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, confidences, conf_threshold_, nms_threshold_,
                      nms_indices);

    for (int idx : nms_indices) {
      adas::core::DetectedObject obj;
      obj.bounding_box = cv::Rect2f(
          static_cast<float>(bboxes[idx].x), static_cast<float>(bboxes[idx].y),
          static_cast<float>(bboxes[idx].width),
          static_cast<float>(bboxes[idx].height));
      obj.confidence = confidences[idx];
      int cid = class_ids[idx];
      obj.class_id = cid;
      if (cid >= 0 && cid < static_cast<int>(active_classes.size())) {
        obj.label = active_classes[cid];
      } else {
        obj.label = "class_" + std::to_string(cid);
      }
      detections.push_back(obj);
    }
  } catch (const cv::Exception &e) {
    std::cerr << "[OpenCVDNNEngine] OpenCV Exception during infer: "
              << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "[OpenCVDNNEngine] Exception during infer: " << e.what()
              << std::endl;
  }

  return detections;
}

bool OpenCVDNNEngine::runLiveDemo(const adas::config::PerceptionConfig &config) {
  cv::VideoCapture cap(config.webcam_id);
  if (!cap.isOpened()) {
    std::cerr << "[OpenCVDNNEngine] Error: Could not open webcam with ID "
              << config.webcam_id << std::endl;
    return false;
  }

  cv::Mat camera_matrix, dist_coeffs;
  int calib_w = 0;
  int calib_h = 0;
  double fx = 1056.46136;
  double cx = 1017.33354;
  double fy = 1063.57938;
  double cy = 535.71139;

  if (!config.intrinsics_path.empty()) {
    cv::FileStorage fs(config.intrinsics_path, cv::FileStorage::READ);
    if (fs.isOpened()) {
      fs["camera_matrix"] >> camera_matrix;
      fs["distortion_coefficients"] >> dist_coeffs;
      fs["image_width"] >> calib_w;
      fs["image_height"] >> calib_h;
      if (!camera_matrix.empty()) {
        fx = camera_matrix.at<double>(0, 0);
        cx = camera_matrix.at<double>(0, 2);
        fy = camera_matrix.at<double>(1, 1);
        cy = camera_matrix.at<double>(1, 2);
        std::cout << "[OpenCVDNNEngine] Loaded camera intrinsics from: "
                  << config.intrinsics_path << std::endl;
      } else {
        std::cerr << "[OpenCVDNNEngine] Warning: Failed to read camera_matrix from: "
                  << config.intrinsics_path << std::endl;
      }
    } else {
      std::cerr << "[OpenCVDNNEngine] Warning: Could not open intrinsics file: "
                << config.intrinsics_path << std::endl;
    }
  }

  // If calibration dimensions are known, request camera resolution to match
  if (calib_w > 0 && calib_h > 0) {
    cap.set(cv::CAP_PROP_FRAME_WIDTH, calib_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, calib_h);
  }

  std::cout << "[OpenCVDNNEngine] Starting live webcam demo on device "
            << config.webcam_id << ". Press 'q' or 'ESC' to exit." << std::endl;

  const std::string window_name = "ADAS YOLO Live Demo";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  cv::Mat raw_frame, frame;
  while (true) {
    cap >> raw_frame;
    if (raw_frame.empty()) {
      std::cerr << "[OpenCVDNNEngine] Warning: Captured empty frame."
                << std::endl;
      break;
    }

    // Adapt camera intrinsics if current frame resolution differs from calibration resolution
    double cur_fx = fx;
    double cur_cx = cx;
    double cur_fy = fy;
    double cur_cy = cy;
    cv::Mat scaled_camera_matrix;

    if (calib_w > 0 && calib_h > 0 &&
        (raw_frame.cols != calib_w || raw_frame.rows != calib_h)) {
      double sx = static_cast<double>(raw_frame.cols) / calib_w;
      double sy = static_cast<double>(raw_frame.rows) / calib_h;
      cur_fx = fx * sx;
      cur_cx = cx * sx;
      cur_fy = fy * sy;
      cur_cy = cy * sy;
      if (!camera_matrix.empty()) {
        scaled_camera_matrix = camera_matrix.clone();
        scaled_camera_matrix.at<double>(0, 0) *= sx;
        scaled_camera_matrix.at<double>(0, 2) *= sx;
        scaled_camera_matrix.at<double>(1, 1) *= sy;
        scaled_camera_matrix.at<double>(1, 2) *= sy;
      }
    } else if (!camera_matrix.empty()) {
      scaled_camera_matrix = camera_matrix.clone();
    } else {
      // Default fallback matrix if no calibration file is provided
      cur_fx = 0.8 * raw_frame.cols;
      cur_fy = 0.8 * raw_frame.cols;
      cur_cx = raw_frame.cols / 2.0;
      cur_cy = raw_frame.rows / 2.0;
      scaled_camera_matrix = (cv::Mat_<double>(3, 3) << cur_fx, 0, cur_cx,
                                                        0, cur_fy, cur_cy,
                                                        0, 0, 1);
    }

    // Undistort frame if distortion coefficients are present
    if (!scaled_camera_matrix.empty() && !dist_coeffs.empty()) {
      cv::undistort(raw_frame, frame, scaled_camera_matrix, dist_coeffs);
    } else {
      frame = raw_frame;
    }

    auto detections = infer(frame);

    size_t rendered_count = 0;
    for (const auto &obj : detections) {
      // Filter out detections not present in target_class_ids (if specified)
      if (!config.target_class_ids.empty()) {
        if (std::find(config.target_class_ids.begin(), config.target_class_ids.end(),
                      obj.class_id) == config.target_class_ids.end()) {
          continue;
        }
      }

      // Determine real physical height in meters for pinhole depth deprojection
      double real_height = config.object_real_height;
      if (config.class_real_heights.count(obj.class_id)) {
        real_height = config.class_real_heights.at(obj.class_id);
      } else {
        // COCO default physical object heights in meters
        switch (obj.class_id) {
          case 0:  real_height = 1.70; break; // person
          case 1:  real_height = 1.00; break; // bicycle
          case 2:  real_height = 1.50; break; // car
          case 3:  real_height = 1.20; break; // motorcycle
          case 5:  real_height = 3.20; break; // bus
          case 7:  real_height = 3.00; break; // truck
          case 9:  real_height = 0.80; break; // traffic light
          case 11: real_height = 0.75; break; // stop sign
          case 39: real_height = 0.25; break; // bottle
          case 41: real_height = 0.10; break; // cup
          default:
            real_height = config.object_real_height;
            break;
        }
      }

      // Run depth deprojection and 3D pose calculation using RoadObject
      adas::perception::RoadObject road_obj(obj, real_height, cur_fy, cur_fx, cur_cx, cur_cy);
      adas::core::TrackedObject tracked = road_obj.process();

      // Bounding box
      cv::Rect bbox_int(
          static_cast<int>(obj.bounding_box.x),
          static_cast<int>(obj.bounding_box.y),
          static_cast<int>(obj.bounding_box.width),
          static_cast<int>(obj.bounding_box.height));
      cv::rectangle(frame, bbox_int, cv::Scalar(0, 255, 0), 2);

      // Format text overlays
      std::string line1 = cv::format("%s %.2f | Depth: %.2fm",
                                     obj.label.c_str(), obj.confidence, tracked.position.x);
      std::string line2 = cv::format("Pose: [X=%.2f, Y=%.2f, Z=%.2f]m",
                                     tracked.position.y, tracked.position.z, tracked.position.x);

      int font = cv::FONT_HERSHEY_SIMPLEX;
      double scale = 0.45;
      int thick = 1;
      int b1 = 0, b2 = 0;
      cv::Size s1 = cv::getTextSize(line1, font, scale, thick, &b1);
      cv::Size s2 = cv::getTextSize(line2, font, scale, thick, &b2);
      int banner_w = std::max(s1.width, s2.width) + 8;
      int banner_h = s1.height + s2.height + 8;

      int banner_x = std::clamp(bbox_int.x, 0, std::max(0, frame.cols - banner_w));
      int banner_y = std::max(banner_h + 2, bbox_int.y);

      // Render solid background banner for high-contrast readability
      cv::rectangle(frame,
                    cv::Point(banner_x, banner_y - banner_h),
                    cv::Point(banner_x + banner_w, banner_y),
                    cv::Scalar(25, 25, 25), cv::FILLED);

      // Draw Depth & Label line (bright green)
      cv::putText(frame, line1,
                  cv::Point(banner_x + 4, banner_y - s2.height - 4),
                  font, scale, cv::Scalar(0, 255, 0), thick);

      // Draw 3D Coordinates line (yellow/cyan)
      cv::putText(frame, line2,
                  cv::Point(banner_x + 4, banner_y - 2),
                  font, scale, cv::Scalar(0, 255, 255), thick);

      rendered_count++;
    }

    // Display HUD at top of screen
    std::string hud = cv::format("Tracked Objects: %zu (Filtered from %zu detections) | 'q'/ESC to exit",
                                 rendered_count, detections.size());
    cv::putText(frame, hud, cv::Point(15, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);

    cv::imshow(window_name, frame);

    int key = cv::waitKey(30);
    if (key == 27 || key == 'q' || key == 'Q') {
      std::cout << "[OpenCVDNNEngine] Live demo stopped by user." << std::endl;
      break;
    }
  }

  cv::destroyWindow(window_name);
  return true;
}

bool OpenCVDNNEngine::runLiveDemo(int webcam_id,
                                 const std::string &intrinsics_path) {
  adas::config::PerceptionConfig config;
  config.webcam_id = webcam_id;
  config.intrinsics_path = intrinsics_path;
  return runLiveDemo(config);
}

} // namespace adas::yolo
