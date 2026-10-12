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

    std::cout << "[Test] ALL TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
