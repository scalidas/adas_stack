#pragma once

#include "core/core_types.hpp"
#include <nlohmann/json.hpp>
#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif
#ifndef _WEBSOCKETPP_CPP11_STL_
#define _WEBSOCKETPP_CPP11_STL_
#endif

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace adas::core {

void to_json(nlohmann::json &j, const Vector3D &v);
void to_json(nlohmann::json &j, const TrackedObject &obj);
void to_json(nlohmann::json &j, const TrafficSign &sign);
void to_json(nlohmann::json &j, const PolynomialLaneBoundary &poly);
void to_json(nlohmann::json &j, const Lane &lane);
void to_json(nlohmann::json &j, const EgoState &ego);
void to_json(nlohmann::json &j, const WorldModel &wm);

} // namespace adas::core

namespace adas::streaming {

struct StreamerConfig {
  uint16_t port{9002};
  std::string host{"0.0.0.0"};
  bool enable_logging{false};
  size_t max_queue_size{100};
  std::string endpoint_path{"/world_model"};
};

class WebSocketStreamer {
public:
  using Server = websocketpp::server<websocketpp::config::asio_no_tls>;
  using ConnectionHdl = websocketpp::connection_hdl;
  using MessagePtr = Server::message_ptr;
  using ConnectionSet = std::set<ConnectionHdl, std::owner_less<ConnectionHdl>>;

  // Constructor & Destructor
  explicit WebSocketStreamer(const StreamerConfig &config = StreamerConfig{});
  ~WebSocketStreamer();

  // Prevent copying
  WebSocketStreamer(const WebSocketStreamer &) = delete;
  WebSocketStreamer &operator=(const WebSocketStreamer &) = delete;

  // Server Lifecycle Management
  bool start();
  void stop();
  bool is_running() const;

  // Data Ingestion & Broadcast Interface
  void publish(const adas::core::WorldModel &world_model);
  nlohmann::json
  serialize_world_model(const adas::core::WorldModel &world_model) const;
  void broadcast_json(const nlohmann::json &j);
  void broadcast_string(const std::string &message);

  // WebSocket Event Handlers
  void on_open(ConnectionHdl hdl);
  void on_close(ConnectionHdl hdl);
  void on_message(ConnectionHdl hdl, MessagePtr msg);

  // Diagnostics & Getters
  size_t get_active_client_count() const;
  uint16_t get_port() const;
  std::string get_host() const;
  const StreamerConfig &get_config() const;

private:
  void init_server();

  StreamerConfig config_;
  Server server_;

  ConnectionSet connections_;
  mutable std::mutex connections_mutex_;

  std::thread server_thread_;
  std::atomic<bool> is_running_{false};
};

} // namespace adas::streaming
