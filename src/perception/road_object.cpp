#include "perception/road_object.hpp"
#include <iomanip>
#include <iostream>
#include "calibration/extrinsic_calibration.hpp"
#include "config/config.hpp"


namespace adas::perception {

RoadObject::RoadObject(const adas::core::DetectedObject& detected_obj,
                       double real_height_m, double fy, double fx, double cx,
                       double cy, const cv::Mat& ground_transform)
    : detected_obj_(detected_obj),
      real_height_m_(real_height_m),
      fy_(fy),
      fx_(fx),
      cx_(cx),
      cy_(cy),
      ground_transform_(ground_transform) {}

RoadObject::RoadObject(const adas::core::DetectedObject& detected_obj,
                       double real_height_m, const cv::Mat& camera_matrix,
                       const cv::Mat& ground_transform)
    : detected_obj_(detected_obj),
      real_height_m_(real_height_m),
      ground_transform_(ground_transform) {
    set_intrinsics(camera_matrix);
}

void RoadObject::set_intrinsics(double fy, double fx, double cx, double cy) {
    fy_ = fy;
    fx_ = fx;
    cx_ = cx;
    cy_ = cy;
}

void RoadObject::set_intrinsics(const cv::Mat& camera_matrix) {
    if (!camera_matrix.empty() && camera_matrix.rows >= 3 &&
        camera_matrix.cols >= 3) {
        fx_ = camera_matrix.at<double>(0, 0);
        cx_ = camera_matrix.at<double>(0, 2);
        fy_ = camera_matrix.at<double>(1, 1);
        cy_ = camera_matrix.at<double>(1, 2);
    }
}

void RoadObject::set_ground_transform(const cv::Mat& ground_transform) {
    ground_transform_ = ground_transform;
}

void RoadObject::set_real_height(double real_height_m) {
    real_height_m_ = real_height_m;
}

adas::core::TrackedObject RoadObject::process() const {
    adas::core::TrackedObject tracked;
    tracked.label = detected_obj_.label;
    tracked.confidence = detected_obj_.confidence;

    float h_pixels = detected_obj_.bounding_box.height;
    if (h_pixels <= 0.0f) {
        std::cerr << "[RoadObject] Invalid bounding box height: " << h_pixels
                  << std::endl;
        return tracked;
    }

    // Pinhole camera depth equation: Z = (fy * H_real) / h_pixels
    double z_depth = (fy_ * real_height_m_) / static_cast<double>(h_pixels);

    // Bounding box center coordinates
    double u_center =
        detected_obj_.bounding_box.x + detected_obj_.bounding_box.width / 2.0;
    double v_center =
        detected_obj_.bounding_box.y + detected_obj_.bounding_box.height / 2.0;

    // Horizontal lateral offset X = (u_center - cx) * Z / fx
    double x_lat = ((u_center - cx_) * z_depth) / fx_;

    // Vertical height offset Y = (cy - v_center) * Z / fy
    double y_vert = ((cy_ - v_center) * z_depth) / fy_;

    // Camera coordinate frame coordinates:
    // x_depth = forward depth along optical axis
    // y_lat   = lateral offset right
    // z_vert  = vertical offset up
    double x_depth = z_depth;

    if (!ground_transform_.empty() && ground_transform_.rows == 3 &&
        ground_transform_.cols == 3) {
        cv::Mat M;
        if (ground_transform_.type() != CV_64F) {
            ground_transform_.convertTo(M, CV_64F);
        } else {
            M = ground_transform_;
        }

        // Transform from camera coordinate frame [x_depth, y_lat, z_vert] to
        // ground plane reference frame:
        // +x axis = forward along the ground
        // +y axis = lateral offset right along the ground
        // +z axis = vertical offset up perpendicular to the ground
        tracked.position.x = M.at<double>(0, 0) * x_depth +
                             M.at<double>(0, 1) * x_lat +
                             M.at<double>(0, 2) * y_vert;
        tracked.position.y = M.at<double>(1, 0) * x_depth +
                             M.at<double>(1, 1) * x_lat +
                             M.at<double>(1, 2) * y_vert;
        tracked.position.z = M.at<double>(2, 0) * x_depth +
                             M.at<double>(2, 1) * x_lat +
                             M.at<double>(2, 2) * y_vert;
    } else {
        // Fallback: camera optical axis coordinate frame
        tracked.position.x = x_depth;
        tracked.position.y = x_lat;
        tracked.position.z = y_vert;
    }

    return tracked;
}

int run_road_object_demo(const std::string& config_path) {
    adas::config::PerceptionConfig config =
        adas::config::load_config(config_path);

    // Default EMeet camera intrinsics (1920x1080)
    double fy = 1063.57938;
    double fx = 1056.46136;
    double cx = 1017.33354;
    double cy = 535.71139;
    int calib_w = 1920;
    int calib_h = 1080;

    // Load camera intrinsics file if provided
    if (!config.intrinsics_path.empty()) {
        cv::FileStorage fs(config.intrinsics_path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            cv::Mat K;
            fs["camera_matrix"] >> K;
            if (!K.empty()) {
                fx = K.at<double>(0, 0);
                cx = K.at<double>(0, 2);
                fy = K.at<double>(1, 1);
                cy = K.at<double>(1, 2);
            }
            fs["image_width"] >> calib_w;
            fs["image_height"] >> calib_h;
            std::cout << "[RoadObject Demo] Loaded camera intrinsics from: "
                      << config.intrinsics_path << std::endl;
        }
    }

    // Load ground transform from extrinsic calibration if provided
    cv::Mat ground_transform;
    if (!config.extrinsics_path.empty()) {
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        adas::calibration::compute_camera_to_ground_transform(
            config.extrinsics_path, K, cv::Mat(), ground_transform, calib_w,
            calib_h);
    }

    // Load image if available
    cv::Mat image = cv::imread(config.image_path);
    if (image.empty()) {
        std::cout << "[RoadObject Demo] Could not open image: "
                  << config.image_path << ". Creating empty 1920x1080 canvas."
                  << std::endl;
        image = cv::Mat::zeros(1080, 1920, CV_8UC3);
    }

    // Scale camera intrinsics if current image size differs from calibration size
    if (calib_w > 0 && calib_h > 0 &&
        (image.cols != calib_w || image.rows != calib_h)) {
        double scale_x = static_cast<double>(image.cols) / calib_w;
        double scale_y = static_cast<double>(image.rows) / calib_h;
        fx *= scale_x;
        cx *= scale_x;
        fy *= scale_y;
        cy *= scale_y;
    }

    // Create DetectedObject from config parameters
    adas::core::DetectedObject detected;
    detected.label = config.object_label;
    detected.bounding_box =
        cv::Rect2f(config.object_bbox_x, config.object_bbox_y,
                   config.object_bbox_width, config.object_bbox_height);
    detected.confidence = 0.95f;

    // Instantiate RoadObject and compute 3D pose in ground frame
    RoadObject road_obj(detected, config.object_real_height, fy, fx, cx, cy,
                        ground_transform);
    adas::core::TrackedObject tracked = road_obj.process();

    // Print results
    std::cout << "\n=================================================="
              << std::endl;
    std::cout << "           RoadObject 3D Pose Demo Results" << std::endl;
    std::cout << "=================================================="
              << std::endl;
    std::cout << "Object Label:          " << tracked.label << std::endl;
    std::cout << "Bounding Box:          [x=" << detected.bounding_box.x
              << ", y=" << detected.bounding_box.y
              << ", w=" << detected.bounding_box.width
              << ", h=" << detected.bounding_box.height << "]" << std::endl;
    std::cout << "Physical Real Height:  " << config.object_real_height << " m"
              << std::endl;
    std::cout << "Camera Focal Length:   fy=" << std::fixed
              << std::setprecision(2) << fy << " px, fx=" << fx << " px"
              << std::endl;
    std::cout << "Camera Principal Pt:   cx=" << cx << " px, cy=" << cy << " px"
              << std::endl;
    std::cout << "--------------------------------------------------"
              << std::endl;
    std::cout << "Calculated 3D Position (Ground Reference Frame):"
              << std::endl;
    std::cout << "  Forward along Ground (+X): " << tracked.position.x << " m"
              << std::endl;
    std::cout << "  Lateral Offset Right (+Y): " << tracked.position.y << " m"
              << std::endl;
    std::cout << "  Vertical Height Up   (+Z): " << tracked.position.z << " m"
              << std::endl;
    std::cout << "Confidence:            " << tracked.confidence << std::endl;
    std::cout << "==================================================\n"
              << std::endl;

    // Visualize on image
    cv::Mat vis = image.clone();
    cv::Rect bbox_int(static_cast<int>(detected.bounding_box.x),
                      static_cast<int>(detected.bounding_box.y),
                      static_cast<int>(detected.bounding_box.width),
                      static_cast<int>(detected.bounding_box.height));
    cv::rectangle(vis, bbox_int, cv::Scalar(0, 255, 0), 2);

    std::string label_str =
        tracked.label + " " + cv::format("%.2fm", tracked.position.x);
    cv::putText(vis, label_str,
                cv::Point(bbox_int.x, std::max(0, bbox_int.y - 10)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    std::string pose_str =
        cv::format("Ground Pose: [Fwd X=%.2fm, Lat Y=%.2fm, Up Z=%.2fm]",
                   tracked.position.x, tracked.position.y, tracked.position.z);
    cv::putText(vis, pose_str, cv::Point(20, vis.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

    cv::imshow("RoadObject 3D Pose Demo", vis);
    cv::waitKey(0);

    return 0;
}

}  // namespace adas::perception