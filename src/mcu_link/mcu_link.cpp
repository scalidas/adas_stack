#include "core/core_types.hpp"
#include "mcu_link/spi_protocol.h"

#ifdef ADAS_ENABLE_MCU_LINK

#include "mcu_link/mcu_link.h"
#include "config/config.hpp"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef __linux__
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// Send one 10-byte binary transaction with one CS assertion.
static bool send_command(int fd, SpiCommand *cmd) {
    if (!spi_command_test_ranges_ok(cmd)) {
        std::fprintf(stderr, "[MCU Link] Command out of range: effort=%d, steer=%d, brake=%d\n",
                     cmd->target_motor_effort, cmd->target_steering_pulse_us, cmd->emergency_brake);
        return false;
    }

#ifdef __linux__
    uint8_t tx[SPI_CMD_BYTES] = {};
    uint8_t unused_rx[SPI_CMD_BYTES] = {};
    spi_command_encode(cmd, tx);

    spi_ioc_transfer transfer = {};
    transfer.tx_buf = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tx));
    transfer.rx_buf = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(unused_rx));
    transfer.len = SPI_CMD_BYTES;
    transfer.speed_hz = SPI_CMD_CLOCK_HZ;
    transfer.bits_per_word = 8;
    transfer.cs_change = 0;

    const int result = ioctl(fd, SPI_IOC_MESSAGE(1), &transfer);
    if (result < 0) {
        std::perror("[MCU Link] SPI transfer failed");
        return false;
    }
    if (result != static_cast<int>(SPI_CMD_BYTES)) {
        std::fprintf(stderr, "[MCU Link] Incomplete SPI transfer: %d bytes\n", result);
        return false;
    }
    return true;
#else
    (void)fd;
    return false;
#endif
}

adas::mcu_link::McuLink::McuLink(const adas::config::PerceptionConfig& cfg) {
    enabled = cfg.mcu_link_enabled;
    spi_device = cfg.mcu_link_device;

    std::cout << "MCU Link Enabled : " << enabled << "\n";


#ifdef __linux__
    fd = static_cast<int>(open(spi_device.c_str(), O_RDWR));
    if (fd < 0) {
        std::perror("[MCU Link] Cannot open SPI");
        enabled = false;
        return;
    }

    // Match RP2350 slave: SPI Mode 1, 8-bit words, MSB first.
    uint8_t mode = SPI_MODE_1;
    uint8_t bits = 8;
    uint8_t lsb_first = 0;
    uint32_t speed_hz = SPI_CMD_CLOCK_HZ;

    if (ioctl(fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ioctl(fd, SPI_IOC_WR_LSB_FIRST, &lsb_first) < 0 ||
        ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed_hz) < 0) {
        std::perror("[MCU Link] Cannot configure SPI");
    }
#else
    std::fprintf(stderr, "[MCU Link] Warning: SPI is only supported on Linux platform.\n");
    enabled = false;
#endif
}

bool adas::mcu_link::McuLink::send(const adas::core::VehicleControlCommand& cmd) {
    if (!enabled) {
        return false;
    }

    static int sequence = 0;

    SpiCommand spi_command;
    spi_command.sequence = static_cast<uint16_t>(sequence++);

    // Motor effort in [0, 100]
    int16_t effort = static_cast<int16_t>(std::clamp(std::round(cmd.target_velocity * 20.0), 0.0, 100.0));
    spi_command.target_motor_effort = effort;

    // Convert steering angle (radians) to pulse microseconds [1000, 2000], center 1500
    // If input is already in [1000, 2000], use directly
    int16_t pulse = 1500;
    if (cmd.target_steering_angle >= 500.0) {
        pulse = static_cast<int16_t>(std::clamp(cmd.target_steering_angle, 1000.0, 2000.0));
    } else {
        // Map [-0.5236, +0.5236] rad -> [1000, 2000] us
        constexpr double MAX_STEER_RAD = 0.5235987755982988;
        double normalized = std::clamp(cmd.target_steering_angle / MAX_STEER_RAD, -1.0, 1.0);
        pulse = static_cast<int16_t>(std::clamp(std::round(1500.0 + normalized * 500.0), 1000.0, 2000.0));
    }
    spi_command.target_steering_pulse_us = pulse;
    spi_command.emergency_brake = cmd.emergency_brake ? 1u : 0u;

    return send_command(fd, &spi_command);
}

#endif