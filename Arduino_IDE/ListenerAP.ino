#include <Arduino.h>
#include <WiFi.h>

extern "C" {
  #include "esp_wifi.h"
}

// -------------------- CONFIG --------------------
#define BAUD_RATE      912600

#define WIFI_CHANNEL   6
#define AP_SSID        "CSI_PRIVATE_AP"
#define AP_PASS        "12345678"   // >= 8 chars

// If you want to reduce serial bandwidth, cap CSI bytes emitted.
// Leave at 256 for full ESP32 capture when csi_len=256.
#define CSI_MAX_BYTES  256

// Optional: restrict CSI to only associated stations (generally true in AP mode anyway)
static const bool FILTER_ONE_STA = false;
static const uint8_t TARGET_STA_MAC[6] = {0x68, 0xFE, 0x71, 0x90, 0x60, 0xA0};

// -------------------- UTIL --------------------
static inline bool mac_eq(const uint8_t *a, const uint8_t *b) {
  for (int i = 0; i < 6; i++) if (a[i] != b[i]) return false;
  return true;
}

// CRC16-CCITT (0x1021) initial 0xFFFF
static uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; b++) {
      if (crc & 0x8000) crc = (crc << 1) ^ 0x1021;
      else crc <<= 1;
    }
  }
  return crc;
}

// Write little-endian helpers
static inline void write_u16_le(uint16_t v) {
  Serial.write((uint8_t)(v & 0xFF));
  Serial.write((uint8_t)((v >> 8) & 0xFF));
}
static inline void write_u32_le(uint32_t v) {
  Serial.write((uint8_t)(v & 0xFF));
  Serial.write((uint8_t)((v >> 8) & 0xFF));
  Serial.write((uint8_t)((v >> 16) & 0xFF));
  Serial.write((uint8_t)((v >> 24) & 0xFF));
}

// -------------------- CSI --------------------
static void emit_csi_frame(const wifi_csi_info_t *info) {
  // Build header in a small buffer so we can CRC it + CSI bytes
  // Header fields after magic:
  // ver, flags, ms(u32), rssi(i8), mac[6], csi_len(u16)
  uint8_t header[1 + 1 + 4 + 1 + 6 + 2];
  size_t idx = 0;

  header[idx++] = 1; // ver
  header[idx++] = 0; // flags
  uint32_t ms = millis();
  header[idx++] = (uint8_t)(ms & 0xFF);
  header[idx++] = (uint8_t)((ms >> 8) & 0xFF);
  header[idx++] = (uint8_t)((ms >> 16) & 0xFF);
  header[idx++] = (uint8_t)((ms >> 24) & 0xFF);

  int8_t rssi = (int8_t)info->rx_ctrl.rssi;
  header[idx++] = (uint8_t)rssi;

  for (int i = 0; i < 6; i++) header[idx++] = info->mac[i];

  uint16_t csi_len = info->len;
  if (csi_len > CSI_MAX_BYTES) csi_len = CSI_MAX_BYTES;

  header[idx++] = (uint8_t)(csi_len & 0xFF);
  header[idx++] = (uint8_t)((csi_len >> 8) & 0xFF);

  // Compute CRC over: header + CSI bytes
  uint16_t crc = crc16_ccitt(header, sizeof(header));
  crc = crc16_ccitt((const uint8_t*)info->buf, csi_len) ^ crc;

  // Emit frame: magic + header + csi + crc
  Serial.write(0xAA);
  Serial.write(0x55);
  Serial.write(header, sizeof(header));
  Serial.write((const uint8_t*)info->buf, csi_len);
  write_u16_le(crc);
}

void on_csi(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;

  if (FILTER_ONE_STA && !mac_eq(info->mac, TARGET_STA_MAC)) return;

  emit_csi_frame(info);
}

static void init_csi() {
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&on_csi, NULL));

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = 1;
  cfg.htltf_en = 1;
  cfg.stbc_htltf2_en = 1;
  cfg.ltf_merge_en = 1;
  cfg.channel_filter_en = 1;
  cfg.manu_scale = 0;
  cfg.shift = 0;

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(BAUD_RATE);
  delay(200);

  WiFi.mode(WIFI_AP);

  bool ok = WiFi.softAP(AP_SSID, AP_PASS, WIFI_CHANNEL, /*hidden=*/false, /*max_conn=*/8);
  if (!ok) {
    Serial.println("AP_START_FAILED");
    return;
  }

  ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

  // One small ASCII line at boot. After that it's binary frames starting with 0xAA55.
  Serial.println("LISTENER_AP_READY");

  init_csi();
}

void loop() {
  delay(1000);
}