#include "yolo/opencv_dnn_engine.hpp"
#include "perception/perceptionengine.hpp"
#include "streaming/websocket_streamer.hpp"
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

    // 10. Test PerceptionConfig webserver deserialization & WorldModel JSON serialization & streaming
    {
      nlohmann::json test_json = {
          {"enable_webserver", true},
          {"webserver_port", 9005},
          {"webserver_host", "127.0.0.1"},
          {"webserver_logging", false}
      };
      adas::config::PerceptionConfig ws_config;
      adas::config::from_json(test_json, ws_config);
      assert(ws_config.enable_webserver == true);
      assert(ws_config.webserver_port == 9005);
      assert(ws_config.webserver_host == "127.0.0.1");

      // Verify WorldModel JSON serialization
      adas::core::WorldModel wm;
      wm.timestamp_us = 123456789;
      wm.frame_seq = 42;
      wm.current_lane.lane_id = 1;
      wm.current_lane.lane_width = 0.50;

      adas::core::TrackedObject obj;
      obj.id = 1;
      obj.label = "bottle";
      obj.bounding_box = cv::Rect2f(10.0f, 20.0f, 100.0f, 200.0f);
      obj.position = {1.5, -0.2, 0.1};
      wm.obstacles.push_back(obj);

      nlohmann::json wm_json;
      adas::core::to_json(wm_json, wm);
      assert(wm_json["timestamp_us"] == 123456789);
      assert(wm_json["frame_seq"] == 42);
      assert(wm_json["obstacles"].size() == 1);
      assert(wm_json["obstacles"][0]["label"] == "bottle");
      assert(wm_json["obstacles"][0]["bounding_box"]["x"] == 10.0f);
      assert(wm_json["obstacles"][0]["position"]["x"] == 1.5);
      std::cout << "[Test] WorldModel serialization passed: " << wm_json.dump() << std::endl;

      // Verify WebSocketStreamer lifecycle and broadcast
      adas::streaming::StreamerConfig streamer_cfg;
      streamer_cfg.port = 9005;
      streamer_cfg.host = "127.0.0.1";
      adas::streaming::WebSocketStreamer streamer(streamer_cfg);
      assert(streamer.start());
      assert(streamer.is_running());
      streamer.publish(wm);
      streamer.stop();
      assert(!streamer.is_running());
      std::cout << "[Test] WebSocketStreamer lifecycle and broadcast passed." << std::endl;

      // Verify PerceptionEngine webserver integration
      ws_config.model_path = ""; // Skip loading heavy model for lightweight test
      adas::perception::PerceptionEngine pe(ws_config);
      assert(pe.is_webserver_running());
      pe.broadcast_world_model(wm);
      pe.stop_webserver();
      assert(!pe.is_webserver_running());
      std::cout << "[Test] PerceptionEngine webserver integration passed." << std::endl;
    }

    std::cout << "[Test] ALL TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
