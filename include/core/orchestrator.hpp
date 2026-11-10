#pragma once

#include "config/config.hpp"
#include "controller/stanley_controller.hpp"
#include "perception/perceptionengine.hpp"
#include "planning/planner.hpp"
#include "streaming/websocket_streamer.hpp"

#ifdef ADAS_ENABLE_MCU_LINK
#include "mcu_link/mcu_link.h"
#endif

namespace adas::core {

class Orchestrator {
   private:
    config::PerceptionConfig config;
    perception::PerceptionEngine perception_engine;
    planning::Planner planner;
    controller::StanleyController controller;

    #ifdef ADAS_ENABLE_MCU_LINK
    mcu_link::McuLink mcu_link;
    #endif

    enum State { STOPPED, CRUISE };
    State state;

    // Camera
    int webcam_id;

    // Webserver / Streaming interface
    bool enable_webserver;
    std::unique_ptr<streaming::WebSocketStreamer> streamer;

    bool start_webserver();
    void stop_webserver();
    bool is_webserver_running() const;
    void broadcast_world_model(const adas::core::WorldModel& world_model);
    streaming::WebSocketStreamer* get_streamer() const { return streamer.get(); }

   public:
    explicit Orchestrator(const adas::config::PerceptionConfig& cfg)
        : config(cfg),
          perception_engine(cfg),
          planner(cfg),
          controller(cfg),
          state(CRUISE),
          webcam_id(cfg.webcam_id),
          enable_webserver(cfg.enable_webserver),
          streamer(std::make_unique<streaming::WebSocketStreamer>(cfg)) {
        if (cfg.image_path.rfind("webcam", 0) == 0) {
            try {
                std::string idx_str = cfg.image_path.substr(6);
                size_t first = idx_str.find_first_not_of(" \t");
                if (first != std::string::npos) {
                    webcam_id = std::stoi(idx_str.substr(first));
                }
            } catch (...) {
                // keep config.webcam_id
            }
        }
        if (enable_webserver) {
            start_webserver();
        }
        #ifdef ADAS_ENABLE_MCU_LINK
        mcu_link = mcu_link::McuLink(cfg);
        #endif
    }

    ~Orchestrator() {
        stop_webserver();
    }

    int run();
};

}  // namespace adas::core
