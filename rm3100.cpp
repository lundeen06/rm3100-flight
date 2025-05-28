#include <stdio.h>
#include <cstring>
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "pico/binary_info.h"
#include <cmath>

// Pin definitions for SPI
#define SPI_PORT spi0
#define SPI_SCK  2   // GPIO 2 - SCK
#define SPI_MOSI 3   // GPIO 3 - MOSI
#define SPI_MISO 4   // GPIO 4 - MISO
#define SPI_CS   5   // GPIO 5 - SSN

// Add binary info for picotool
bi_decl(bi_4pins_with_func(SPI_MISO, SPI_MOSI, SPI_SCK, SPI_CS, GPIO_FUNC_SPI));
bi_decl(bi_program_description("RM3100 Magnetometer Reader - SPI Interface"));

// Register Map
#define RM3100_REG_POLL   0x00
#define RM3100_REG_CMM    0x01
#define RM3100_REG_CCX    0x04
#define RM3100_REG_CCY    0x06
#define RM3100_REG_CCZ    0x08
#define RM3100_REG_TMRC   0x0B
#define RM3100_REG_MX     0x24
#define RM3100_REG_MY     0x27
#define RM3100_REG_MZ     0x2A
#define RM3100_REG_STATUS 0x34
#define RM3100_REG_REVID  0x36

#define RM3100_REVID 0x22
#define RM3100_CMM_RATE_75_0_HZ  0x05
#define RM3100_CMM_RATE_MSB      0x90
#define RM3100_DRDM_ALL_AXES 0x02

bool sensor_connected = false;

void spi_cs_select() {
    gpio_put(SPI_CS, 0);
}

void spi_cs_deselect() {
    gpio_put(SPI_CS, 1);
}

bool spi_write_reg(uint8_t reg, const uint8_t* data, size_t len) {
    uint8_t cmd = reg;
    
    spi_cs_select();
    int ret = spi_write_blocking(SPI_PORT, &cmd, 1);
    if (ret != 1) {
        spi_cs_deselect();
        return false;
    }
    ret = spi_write_blocking(SPI_PORT, data, len);
    spi_cs_deselect();
    
    return ret == len;
}

bool spi_read_reg(uint8_t reg, uint8_t* data, size_t len) {
    uint8_t cmd = reg | 0x80;
    
    spi_cs_select();
    int ret = spi_write_blocking(SPI_PORT, &cmd, 1);
    if (ret != 1) {
        spi_cs_deselect();
        return false;
    }
    ret = spi_read_blocking(SPI_PORT, 0, data, len);
    spi_cs_deselect();
    
    return ret == len;
}

bool init_sensor() {
    uint8_t rev;
    if (!spi_read_reg(RM3100_REG_REVID, &rev, 1)) {
        return false;
    }
    
    if (rev != RM3100_REVID) {
        return false;
    }
    
    uint8_t cc_buffer[6] = {0, 200, 0, 200, 0, 200};
    if (!spi_write_reg(RM3100_REG_CCX, cc_buffer, 6)) {
        return false;
    }
    
    uint8_t rate = RM3100_CMM_RATE_75_0_HZ | RM3100_CMM_RATE_MSB;
    if (!spi_write_reg(RM3100_REG_TMRC, &rate, 1)) {
        return false;
    }
    
    uint8_t cmm = (1 << 0) | (RM3100_DRDM_ALL_AXES << 2) | (1 << 4) | (1 << 5) | (1 << 6);
    if (!spi_write_reg(RM3100_REG_CMM, &cmm, 1)) {
        return false;
    }
    
    return true;
}

bool read_sample(float* x, float* y, float* z, float* magnitude) {
    uint8_t status;
    if (!spi_read_reg(RM3100_REG_STATUS, &status, 1)) {
        return false;
    }
    
    if (!(status & 0x80)) {
        return false;
    }
    
    uint8_t data[9];
    if (!spi_read_reg(RM3100_REG_MX, data, 9)) {
        return false;
    }
    
    int32_t raw_x = ((int32_t)data[0] << 16) | ((int32_t)data[1] << 8) | data[2];
    int32_t raw_y = ((int32_t)data[3] << 16) | ((int32_t)data[4] << 8) | data[5];
    int32_t raw_z = ((int32_t)data[6] << 16) | ((int32_t)data[7] << 8) | data[8];
    
    if (raw_x & 0x800000) raw_x |= 0xFF000000;
    if (raw_y & 0x800000) raw_y |= 0xFF000000;
    if (raw_z & 0x800000) raw_z |= 0xFF000000;
    
    const float scale = 1.0f / 75.0f;
    *x = raw_x * scale;
    *y = raw_y * scale;
    *z = raw_z * scale;
    
    *magnitude = std::sqrt((*x * *x) + (*y * *y) + (*z * *z));
    
    return true;
}

int main() {
    stdio_init_all();
    sleep_ms(5000);
    printf("rm3100.cpp running :D\n");
    
    spi_init(SPI_PORT, 1000 * 1000);
    spi_set_format(SPI_PORT, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);
    
    gpio_set_function(SPI_SCK, GPIO_FUNC_SPI);
    gpio_set_function(SPI_MOSI, GPIO_FUNC_SPI);
    gpio_set_function(SPI_MISO, GPIO_FUNC_SPI);
    
    gpio_init(SPI_CS);
    gpio_set_dir(SPI_CS, GPIO_OUT);
    gpio_put(SPI_CS, 1);
    
    printf("Timestamp(ms),M_x(µT),M_y(µT),M_z(µT),M(µT)\n");
    
    while (true) {
        if (!sensor_connected) {
            if (init_sensor()) {
                sensor_connected = true;
            }
        } else {
            float x, y, z, magnitude;
            if (read_sample(&x, &y, &z, &magnitude)) {
                printf("%llu,%.3f,%.3f,%.3f,%.3f\n", 
                    time_us_64() / 1000, x, y, z, magnitude);
            } else {
                sensor_connected = false;
            }
        }
        printf(sensor_connected ? "sensor connected\n" : "sensor not connected\n"); // debugging
        sleep_ms(10);
    }
    
    return 0;
}