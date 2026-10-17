#include "yolo/opencv_dnn_engine.hpp"
#include <iostream>
#include <memory>
#include <cassert>

int main() {
    std::cout << "[Test] Starting OpenCVDNNEngine test suite..." << std::endl;

    // 1. Test polymorphism and virtual destructor deletion via base pointer
    InferenceEngine* base_ptr = new OpenCVDNNEngine();
    assert(base_ptr != nullptr);
    std::cout << "[Test] Created OpenCVDNNEngine through InferenceEngine pointer." << std::endl;
    delete base_ptr;
    std::cout << "[Test] Successfully deleted OpenCVDNNEngine via virtual base destructor." << std::endl;

    // 2. Test unique_ptr management
    std::unique_ptr<InferenceEngine> engine = std::make_unique<OpenCVDNNEngine>();
    assert(!engine->loadModel("nonexistent_model.onnx"));
    std::cout << "[Test] loadModel for non-existent file handled gracefully." << std::endl;

    // 3. Test empty infer handling
    cv::Mat empty_frame;
    auto detections = engine->infer(empty_frame);
    assert(detections.empty());
    std::cout << "[Test] Empty frame inference handled cleanly." << std::endl;

    // 4. Test dummy frame inference when not loaded
    cv::Mat dummy_frame = cv::Mat::zeros(640, 640, CV_8UC3);
    detections = engine->infer(dummy_frame);
    assert(detections.empty());
    std::cout << "[Test] Dummy frame inference without loaded model handled cleanly." << std::endl;

    // 5. Test configuration setters/getters
    OpenCVDNNEngine concrete_engine;
    concrete_engine.setConfidenceThreshold(0.3f);
    concrete_engine.setNMSThreshold(0.5f);
    assert(concrete_engine.getConfidenceThreshold() == 0.3f);
    assert(concrete_engine.getNMSThreshold() == 0.5f);
    std::cout << "[Test] Setter and getter tests passed." << std::endl;

    // 6. Test DetectedObject class_id field and PerceptionConfig target_class_ids
    adas::core::DetectedObject test_obj;
    test_obj.class_id = 2;
    test_obj.label = "car";
    assert(test_obj.class_id == 2);

    adas::config::PerceptionConfig test_config;
    assert(!test_config.target_class_ids.empty());
    std::cout << "[Test] PerceptionConfig target_class_ids and DetectedObject class_id tests passed." << std::endl;

    // 7. Test camera intrinsics configuration and infer populating position field
    concrete_engine.setCameraIntrinsics(1000.0, 1000.0, 320.0, 240.0, 640, 480);
    assert(concrete_engine.getFx() == 1000.0);
    assert(concrete_engine.getFy() == 1000.0);
    assert(concrete_engine.getCx() == 320.0);
    assert(concrete_engine.getCy() == 240.0);
    concrete_engine.setObjectRealHeight(0.2);
    assert(concrete_engine.getObjectRealHeight() == 0.2);
    std::cout << "[Test] Camera intrinsics and object height setters passed." << std::endl;

    // 8. Test infer on real sample image populating obj.position
    OpenCVDNNEngine live_engine;
    if (live_engine.loadModel("models/yolov8n.onnx")) {
      cv::Mat sample = cv::imread("assets/samples/cup.jpg");
      if (!sample.empty()) {
        auto cup_detections = live_engine.infer(sample);
        std::cout << "[Test] Model infer on sample image returned " << cup_detections.size() << " detections." << std::endl;
        for (const auto &d : cup_detections) {
          std::cout << "[Test] Detected: " << d.label << " (id=" << d.class_id
                    << ") Conf=" << d.confidence
                    << " Pos=[Depth=" << d.position.x
                    << ", Lat=" << d.position.y
                    << ", Vert=" << d.position.z << "]" << std::endl;
          if (d.bounding_box.height > 0.0f) {
            assert(d.position.x > 0.0);
          }
        }
      }
    }

    // 9. Test class filtering with setClasses
    if (live_engine.isLoaded()) {
      cv::Mat sample = cv::imread("assets/samples/cup.jpg");
      if (!sample.empty()) {
        // Filter for bottle only (id 39) -> should NOT match cup (id 41)
        live_engine.setClasses(std::vector<int>{39});
        auto filtered_out = live_engine.infer(sample);
        assert(filtered_out.empty());

        // Filter for cup (id 41) -> should match cup
        live_engine.setClasses(std::vector<int>{41});
        auto cup_only = live_engine.infer(sample);
        assert(cup_only.size() == 1);
        assert(cup_only[0].class_id == 41);
        assert(cup_only[0].label == "cup");

        // Clear filter -> should match cup
        live_engine.setClasses(std::vector<int>{});
        auto all_dets = live_engine.infer(sample);
        assert(all_dets.size() == 1);
        assert(all_dets[0].class_id == 41);
        std::cout << "[Test] Target class filtering tests passed." << std::endl;
      }
    }

    std::cout << "[Test] ALL TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
