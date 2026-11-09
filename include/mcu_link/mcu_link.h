#pragma once

#ifdef ADAS_ENABLE_MCU_LINK

#include <string>

#include "config/config.hpp"
#include "core/core_types.hpp"
#include "mcu_link/spi_protocol.h"

namespace adas::mcu_link {
    class McuLink {
        private:
            bool enabled{false};
            std::string spi_device{"/dev/null"}; //Name of SPI device
            int fd{-1}; //File descriptor ID

        public:
            McuLink(adas::config::PerceptionConfig cfg);

            bool send(adas::core::VehicleControlCommand cmd);
    };
}

#endif