#include "calibration/extrinsic_calibration.hpp"
#include <algorithm>
#include <cmath>

namespace adas::calibration {

struct CalibrationState {
  std::vector<cv::Point2f> points;
  int selected_point{-1};
  bool dragging{false};
  cv::Point2f current_mouse_pt{320.0f, 240.0f};
  float zoom_factor{4.0f}; // Default 4x magnification
};

static void mouse_callback(int event, int x, int y, int flags, void *userdata) {
  (void)flags;
  auto *state = static_cast<CalibrationState *>(userdata);
  cv::Point2f mouse_pt(static_cast<float>(x), static_cast<float>(y));
  state->current_mouse_pt = mouse_pt;

  if (event == cv::EVENT_LBUTTONDOWN) {
    // Find nearest point of the four (tl, tr, bl, br) to the click
    float min_dist = 20.0f; // Must be within 20px to register
    state->selected_point = -1;
    for (size_t i = 0; i < state->points.size(); ++i) {
      float dist = static_cast<float>(cv::norm(state->points[i] - mouse_pt));
      if (dist < min_dist) {
        min_dist = dist;
        state->selected_point = static_cast<int>(i);
      }
    }
    if (state->selected_point != -1) {
      state->dragging = true;
      state->points[state->selected_point] = mouse_pt;
    }
  } else if (event == cv::EVENT_MOUSEMOVE && state->dragging) {
    if (state->selected_point >= 0 &&
        state->selected_point < static_cast<int>(state->points.size())) {
      state->points[state->selected_point] = mouse_pt;
    }
  } else if (event == cv::EVENT_LBUTTONUP) {
    state->dragging = false;
    state->selected_point = -1;
  }
}

bool save_extrinsic_calibration(const std::string &filename,
                                const std::vector<cv::Point2f> &src_pts,
                                const std::vector<cv::Point2f> &dst_pts,
                                const cv::Mat &H) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "[Extrinsics] Failed to open file for writing: " << filename
              << std::endl;
    return false;
  }

  cv::Mat src_mat(src_pts), dst_mat(dst_pts);
  fs << "homography_matrix" << H;
  fs << "src_points" << src_mat;
  fs << "dst_points" << dst_mat;

  std::cout << "[Extrinsics] Successfully saved extrinsic calibration to: "
            << filename << std::endl;
  return true;
}

bool load_extrinsic_calibration(const std::string &filename, cv::Mat &H,
                                std::vector<cv::Point2f> *src_pts,
                                std::vector<cv::Point2f> *dst_pts) {
  if (filename.empty())
    return false;

  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  fs["homography_matrix"] >> H;
  if (H.empty()) {
    std::cerr << "[Extrinsics] Invalid homography matrix in: " << filename
              << std::endl;
    return false;
  }

  if (src_pts) {
    cv::Mat src_mat;
    fs["src_points"] >> src_mat;
    if (!src_mat.empty()) {
      src_pts->clear();
      for (int i = 0; i < src_mat.rows; ++i) {
        src_pts->push_back(src_mat.at<cv::Point2f>(i));
      }
    }
  }

  if (dst_pts) {
    cv::Mat dst_mat;
    fs["dst_points"] >> dst_mat;
    if (!dst_mat.empty()) {
      dst_pts->clear();
      for (int i = 0; i < dst_mat.rows; ++i) {
        dst_pts->push_back(dst_mat.at<cv::Point2f>(i));
      }
    }
  }

  std::cout << "[Extrinsics] Successfully loaded extrinsic calibration from: "
            << filename << std::endl;
  return true;
}

int run_extrinsic_calibration(const std::string &image_path,
                              const std::string &output_path) {
  cv::Mat frame = cv::imread(image_path);
  if (frame.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return -1;
  }

  cv::resize(frame, frame, cv::Size(640, 480));

  CalibrationState state;
  cv::Mat dummy_H;
  // Try to load existing points from file if available
  if (!load_extrinsic_calibration(output_path, dummy_H, &state.points, nullptr) ||
      state.points.size() != 4) {
    // Default 4 corner source points
    state.points = {{223.0f, 240.0f},
                    {391.0f, 241.0f},
                    {170.0f, 393.0f},
                    {444.0f, 394.0f}};
  }

  const std::vector<cv::Point2f> dst_pts = {
      {220.0f, 143.0f}, {420.0f, 143.0f}, {220.0f, 400.0f}, {420.0f, 400.0f}};

  const std::string win_name = "Interactive Calibration";
  cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(win_name, mouse_callback, &state);

  while (true) {
    cv::Mat canvas = frame.clone();

    // Convert points for drawing
    std::vector<cv::Point> int_pts;
    for (const auto &pt : state.points) {
      int_pts.push_back(
          cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    // Connect points
    cv::line(canvas, int_pts[0], int_pts[1], cv::Scalar(0, 255, 0), 2);
    cv::line(canvas, int_pts[1], int_pts[3], cv::Scalar(0, 255, 0), 2);
    cv::line(canvas, int_pts[3], int_pts[2], cv::Scalar(0, 255, 0), 2);
    cv::line(canvas, int_pts[2], int_pts[0], cv::Scalar(0, 255, 0), 2);

    // Labels
    const std::vector<std::string> labels = {"0: TL", "1: TR", "2: BL",
                                             "3: BR"};

    // Draw handles and coordinate text
    for (size_t i = 0; i < state.points.size(); ++i) {
      cv::Scalar color = (static_cast<int>(i) == state.selected_point)
                             ? cv::Scalar(0, 0, 255)
                             : cv::Scalar(0, 255, 255);
      cv::circle(canvas, int_pts[i], 6, color, -1);
      cv::circle(canvas, int_pts[i], 8, cv::Scalar(0, 0, 0), 1);

      std::string text =
          labels[i] + " (" +
          std::to_string(static_cast<int>(state.points[i].x)) + ", " +
          std::to_string(static_cast<int>(state.points[i].y)) + ")";
      cv::putText(canvas, text, int_pts[i] + cv::Point(10, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1,
                  cv::LINE_AA);
    }

    // Picture-in-Picture zoom
    cv::Point2f zoom_target = (state.selected_point != -1)
                                  ? state.points[state.selected_point]
                                  : state.current_mouse_pt;

    const int pip_size = 180;
    int crop_w = static_cast<int>(pip_size / state.zoom_factor);
    int crop_h = static_cast<int>(pip_size / state.zoom_factor);

    int crop_x = std::clamp(static_cast<int>(zoom_target.x) - crop_w / 2, 0,
                            frame.cols - crop_w);
    int crop_y = std::clamp(static_cast<int>(zoom_target.y) - crop_h / 2, 0,
                            frame.rows - crop_h);

    cv::Rect crop_rect(crop_x, crop_y, crop_w, crop_h);
    cv::Mat cropped = frame(crop_rect);
    cv::Mat zoomed;
    cv::resize(cropped, zoomed, cv::Size(pip_size, pip_size), 0, 0,
               cv::INTER_NEAREST);

    // Crosshair for zoom
    int center = pip_size / 2;
    cv::line(zoomed, cv::Point(center - 15, center),
             cv::Point(center + 15, center), cv::Scalar(0, 0, 255), 1);
    cv::line(zoomed, cv::Point(center, center - 15),
             cv::Point(center, center + 15), cv::Scalar(0, 0, 255), 1);
    cv::circle(zoomed, cv::Point(center, center), 2, cv::Scalar(0, 255, 255), -1);

    // Move the window out of the way of the mouse
    int pip_x = (zoom_target.x > 320 && zoom_target.y < 240) ? 10 : (canvas.cols - pip_size - 10);
    int pip_y = 10;
    cv::Rect pip_roi(pip_x, pip_y, pip_size, pip_size);
    zoomed.copyTo(canvas(pip_roi));

    // PIP border and label
    cv::rectangle(canvas, pip_roi, cv::Scalar(255, 255, 255), 2);
    std::string zoom_text = "Zoom " + std::to_string(static_cast<int>(state.zoom_factor)) + "x (Scroll)";
    cv::putText(canvas, zoom_text, cv::Point(pip_x + 5, pip_y + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    cv::imshow(win_name, canvas);

    int key = cv::waitKey(30);
    if (key == 27 || key == 's' || key == 'S') { // ESC or 's' key to save and exit
      break;
    } else if (key == '+' || key == '=') {
      state.zoom_factor = std::min(10.0f, state.zoom_factor + 1.0f);
    } else if (key == '-' || key == '_') {
      state.zoom_factor = std::max(2.0f, state.zoom_factor - 1.0f);
    }
  }

  cv::destroyWindow(win_name);

  std::cout << "\nFinal Calibrated Corner Points:\n";
  std::cout << "  Top-Left:     {" << state.points[0].x << "f, "
            << state.points[0].y << "f}\n";
  std::cout << "  Top-Right:    {" << state.points[1].x << "f, "
            << state.points[1].y << "f}\n";
  std::cout << "  Bottom-Left:  {" << state.points[2].x << "f, "
            << state.points[2].y << "f}\n";
  std::cout << "  Bottom-Right: {" << state.points[3].x << "f, "
            << state.points[3].y << "f}\n\n";

  // Compute homography matrix and save to file
  cv::Mat H = cv::getPerspectiveTransform(state.points, dst_pts);
  save_extrinsic_calibration(output_path, state.points, dst_pts, H);

  return 0;
}

} // namespace adas::calibration