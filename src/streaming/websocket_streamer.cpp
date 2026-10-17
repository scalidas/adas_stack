#include "streaming/websocket_streamer.hpp"
#include <iostream>

namespace adas::core {

void to_json(nlohmann::json& j, const Vector3D& v) {
    j = nlohmann::json{{"x", v.x}, {"y", v.y}, {"z", v.z}};
}

void to_json(nlohmann::json& j, const TrackedObject& obj) {
    j = nlohmann::json{{"id", obj.id},
                       {"label", obj.label},
                       {"bounding_box",
                        {{"x", obj.bounding_box.x},
                         {"y", obj.bounding_box.y},
                         {"width", obj.bounding_box.width},
                         {"height", obj.bounding_box.height}}},
                       {"position", obj.position},
                       {"velocity", obj.velocity},
                       {"length", obj.length},
                       {"width", obj.width},
                       {"height", obj.height},
                       {"confidence", obj.confidence}};
}

void to_json(nlohmann::json& j, const TrafficSign& sign) {
    std::string type_str = "UNKNOWN";
    if (sign.type == TrafficSign::Type::STOP_SIGN) {
        type_str = "STOP_SIGN";
    }
    j = nlohmann::json{{"type", type_str},
                       {"position", sign.position},
                       {"confidence", sign.confidence}};
}

void to_json(nlohmann::json& j, const PolynomialLaneBoundary& poly) {
    j = nlohmann::json{{"c0", poly.c0}, {"c1", poly.c1}, {"c2", poly.c2}};
}

void to_json(nlohmann::json& j, const Lane& lane) {
    j = nlohmann::json{{"lane_id", lane.lane_id},
                       {"left_boundary", lane.left_boundary},
                       {"right_boundary", lane.right_boundary},
                       {"centerline", lane.centerline},
                       {"lane_width", lane.lane_width},
                       {"can_change_left", lane.can_change_left},
                       {"can_change_right", lane.can_change_right}};
}

void to_json(nlohmann::json& j, const EgoState& ego) {
    j = nlohmann::json{{"x", ego.x},
                       {"y", ego.y},
                       {"yaw", ego.yaw},
                       {"velocity", ego.velocity},
                       {"steering_angle", ego.steering_angle}};
}

void to_json(nlohmann::json& j, const WorldModel& wm) {
    j = nlohmann::json{{"timestamp_us", wm.timestamp_us},
                       {"frame_seq", wm.frame_seq},
                       {"ego_state", wm.ego_state},
                       {"current_lane", wm.current_lane},
                       {"left_neighbor_lane",
                        wm.left_neighbor_lane.has_value()
                            ? nlohmann::json(wm.left_neighbor_lane.value())
                            : nlohmann::json(nullptr)},
                       {"right_neighbor_lane",
                        wm.right_neighbor_lane.has_value()
                            ? nlohmann::json(wm.right_neighbor_lane.value())
                            : nlohmann::json(nullptr)},
                       {"obstacles", wm.obstacles},
                       {"traffic_signs", wm.traffic_signs}};
}

}  // namespace adas::core

namespace adas::streaming {

WebSocketStreamer::WebSocketStreamer(const StreamerConfig& config)
    : config_(config) {
    init_server();
}

WebSocketStreamer::~WebSocketStreamer() {
    stop();
}

void WebSocketStreamer::init_server() {
    if (!config_.enable_logging) {
        server_.clear_access_channels(websocketpp::log::alevel::all);
        server_.clear_error_channels(websocketpp::log::elevel::all);
    } else {
        server_.set_access_channels(websocketpp::log::alevel::all);
        server_.clear_access_channels(websocketpp::log::alevel::frame_payload);
        server_.set_error_channels(websocketpp::log::elevel::all);
    }

    server_.init_asio();
    server_.set_reuse_addr(true);

    server_.set_open_handler([this](ConnectionHdl hdl) { on_open(hdl); });
    server_.set_close_handler([this](ConnectionHdl hdl) { on_close(hdl); });
    server_.set_message_handler(
        [this](ConnectionHdl hdl, MessagePtr msg) { on_message(hdl, msg); });
}

bool WebSocketStreamer::start() {
    if (is_running_.load()) {
        return true;
    }

    websocketpp::lib::error_code ec;

    server_.reset();

    if (config_.host.empty() || config_.host == "0.0.0.0") {
        server_.listen(config_.port, ec);
    } else {
        server_.listen(config_.host, std::to_string(config_.port), ec);
    }

    if (ec) {
        std::cerr << "[WebSocketStreamer] Listen failed on " << config_.host
                  << ":" << config_.port << " - " << ec.message() << std::endl;
        return false;
    }

    server_.start_accept(ec);
    if (ec) {
        std::cerr << "[WebSocketStreamer] start_accept failed: " << ec.message()
                  << std::endl;
        return false;
    }

    is_running_.store(true);
    server_thread_ = std::thread([this]() {
        try {
            server_.run();
        } catch (const std::exception& e) {
            std::cerr << "[WebSocketStreamer] Exception in server thread: "
                      << e.what() << std::endl;
        }
        is_running_.store(false);
    });

    if (config_.enable_logging) {
        std::cout << "[WebSocketStreamer] Listening on " << config_.host << ":"
                  << config_.port << std::endl;
    }

    return true;
}

void WebSocketStreamer::stop() {
    if (!is_running_.load()) {
        return;
    }

    is_running_.store(false);

    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (const auto& hdl : connections_) {
            websocketpp::lib::error_code ec;
            server_.close(hdl, websocketpp::close::status::going_away,
                          "Server shutting down", ec);
        }
        connections_.clear();
    }

    websocketpp::lib::error_code ec;
    server_.stop_listening(ec);
    server_.stop();

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    if (config_.enable_logging) {
        std::cout << "[WebSocketStreamer] Stopped." << std::endl;
    }
}

bool WebSocketStreamer::is_running() const {
    return is_running_.load();
}

void WebSocketStreamer::publish(const adas::core::WorldModel& world_model) {
    if (!is_running_.load()) {
        return;
    }
    broadcast_json(serialize_world_model(world_model));
}

nlohmann::json WebSocketStreamer::serialize_world_model(
    const adas::core::WorldModel& world_model) const {
    nlohmann::json j;
    adas::core::to_json(j, world_model);
    return j;
}

void WebSocketStreamer::broadcast_json(const nlohmann::json& j) {
    broadcast_string(j.dump());
}

void WebSocketStreamer::broadcast_string(const std::string& message) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (const auto& hdl : connections_) {
        websocketpp::lib::error_code ec;
        server_.send(hdl, message, websocketpp::frame::opcode::text, ec);
        if (ec && config_.enable_logging) {
            std::cerr << "[WebSocketStreamer] Send failed: " << ec.message()
                      << std::endl;
        }
    }
}

void WebSocketStreamer::on_open(ConnectionHdl hdl) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_.insert(hdl);
    if (config_.enable_logging) {
        std::cout << "[WebSocketStreamer] Client connected. Active clients: "
                  << connections_.size() << std::endl;
    }
}

void WebSocketStreamer::on_close(ConnectionHdl hdl) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_.erase(hdl);
    if (config_.enable_logging) {
        std::cout << "[WebSocketStreamer] Client disconnected. Active clients: "
                  << connections_.size() << std::endl;
    }
}

void WebSocketStreamer::on_message(ConnectionHdl /*hdl*/, MessagePtr msg) {
    if (config_.enable_logging && msg) {
        std::cout << "[WebSocketStreamer] Received message: "
                  << msg->get_payload() << std::endl;
    }
}

size_t WebSocketStreamer::get_active_client_count() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    return connections_.size();
}

uint16_t WebSocketStreamer::get_port() const {
    return config_.port;
}

std::string WebSocketStreamer::get_host() const {
    return config_.host;
}

const StreamerConfig& WebSocketStreamer::get_config() const {
    return config_;
}

}  // namespace adas::streaming
