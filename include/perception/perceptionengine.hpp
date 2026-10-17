#include <chrono>
#include <unordered_set>
#include <vector>


#include <opencv2/opencv.hpp>

#include "config/config.hpp"
#include "core/core_types.hpp"
#include "yolo/inference_engine.hpp"
#include "yolo/opencv_dnn_engine.hpp"

namespace adas::streaming {
class WebSocketStreamer;
}

namespace adas::perception {

/*
PerceptionEngine is the single class which will be instantiated in main, and 
actually call all the individual perception algorithms, and build up a cohesive world
model.

It will return a WorldModel representing the current state of perception which
can be passed on to the planning module, and also be converted to JSON and streamed
via websocket
*/
class PerceptionEngine {
   private:
    config::PerceptionConfig config;

    std::unique_ptr<yolo::InferenceEngine> inference_engine;

    std::unordered_set<std::string> target_labels;

    std::chrono::time_point<std::chrono::system_clock> start_time;
    uint32_t frame_counter{0};

    cv::Mat ipm;

    std::unique_ptr<streaming::WebSocketStreamer> streamer;

    //Extract the current lane from the given frame
    std::optional<adas::core::Lane> process_lanes(const cv::Mat& frame);

    //Extract all tracked objects from the given frame
    std::vector<adas::core::TrackedObject> process_obstacles(
        const cv::Mat& frame);

   public:
    PerceptionEngine();
    explicit PerceptionEngine(const adas::config::PerceptionConfig& cfg);
    ~PerceptionEngine();

    adas::core::WorldModel process_frame(const cv::Mat& frame);

    bool runLiveDemo();

    // Webserver / Streaming interface
    bool start_webserver();
    void stop_webserver();
    bool is_webserver_running() const;
    void broadcast_world_model(const adas::core::WorldModel& world_model);
    streaming::WebSocketStreamer* get_streamer() const { return streamer.get(); }
};

}  // namespace adas::perception