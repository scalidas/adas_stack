#include "yolo/opencv_dnn_engine.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

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

bool OpenCVDNNEngine::runLiveDemo(int webcam_id,
                                 const std::string &intrinsics_path) {
  cv::VideoCapture cap(webcam_id);
  if (!cap.isOpened()) {
    std::cerr << "[OpenCVDNNEngine] Error: Could not open webcam with ID "
              << webcam_id << std::endl;
    return false;
  }

  cv::Mat camera_matrix, dist_coeffs;
  if (!intrinsics_path.empty()) {
    cv::FileStorage fs(intrinsics_path, cv::FileStorage::READ);
    if (fs.isOpened()) {
      fs["camera_matrix"] >> camera_matrix;
      fs["distortion_coefficients"] >> dist_coeffs;
      if (!camera_matrix.empty() && !dist_coeffs.empty()) {
        std::cout << "[OpenCVDNNEngine] Loaded camera intrinsics from: "
                  << intrinsics_path << " for frame undistortion." << std::endl;
      } else {
        std::cerr << "[OpenCVDNNEngine] Warning: Failed to read "
                     "camera_matrix/distortion_coefficients from: "
                  << intrinsics_path << std::endl;
      }
    } else {
      std::cerr << "[OpenCVDNNEngine] Warning: Could not open intrinsics file: "
                << intrinsics_path << std::endl;
    }
  }

  std::cout << "[OpenCVDNNEngine] Starting live webcam demo on device "
            << webcam_id << ". Press 'q' or 'ESC' to exit." << std::endl;

  const std::string window_name = "YOLO Live Demo";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  cv::Mat raw_frame, frame;
  while (true) {
    cap >> raw_frame;
    if (raw_frame.empty()) {
      std::cerr << "[OpenCVDNNEngine] Warning: Captured empty frame."
                << std::endl;
      break;
    }

    if (!camera_matrix.empty() && !dist_coeffs.empty()) {
      cv::undistort(raw_frame, frame, camera_matrix, dist_coeffs);
    } else {
      frame = raw_frame;
    }

    auto detections = infer(frame);

    for (const auto &obj : detections) {
      // Draw bounding box
      cv::rectangle(frame, obj.bounding_box, cv::Scalar(0, 255, 0), 2);

      // Create label string
      std::string label = obj.label;
      if (obj.confidence > 0.0f) {
        label += " " + cv::format("%.2f", obj.confidence);
      }

      int baseLine = 0;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

      int top = std::max(static_cast<int>(obj.bounding_box.y),
                         label_size.height);
      int left = static_cast<int>(obj.bounding_box.x);

      cv::rectangle(
          frame, cv::Point(left, top - label_size.height - 2),
          cv::Point(left + label_size.width, top + baseLine),
          cv::Scalar(0, 255, 0), cv::FILLED);

      cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 0, 0), 1);
    }

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

} // namespace adas::yolo
