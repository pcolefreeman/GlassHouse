// ListenerAP.ino
// Captures CSI from connected shouters and emits binary frames over Serial.
// Frame format (no CRC):
//   [0xAA][0x55] magic
//   [1]  ver
//   [1]  flags
//   [4]  timestamp ms (little-endian uint32)
//   [1]  rssi (int8)
//   [1]  noise_floor (int8)
//   [6]  mac
//   [2]  csi_len (little-endian uint16)
//   [N]  csi bytes (N = csi_len)
// Total header after magic: 16 bytes

#include <Arduino.h>
#include <WiFi.h>

extern "C" {
  #include "esp_wifi.h"
}

// -------------------- CONFIG --------------------
#define BAUD_RATE      921600
#define WIFI_CHANNEL   6
#define AP_SSID        "CSI_PRIVATE_AP"
#define AP_PASS        "12345678"

#define CSI_MAX_BYTES  384

// -------------------- CSI --------------------
static void emit_csi_frame(const wifi_csi_info_t *info) {
  // Header layout: ver(1) flags(1) ms(4) rssi(1) noise_floor(1) mac(6) csi_len(2)
  // = 16 bytes total
  uint8_t header[16];
  size_t idx = 0;

  header[idx++] = 1; // ver
  header[idx++] = 0; // flags

  uint32_t ms = millis();
  header[idx++] = (uint8_t)(ms & 0xFF);
  header[idx++] = (uint8_t)((ms >> 8)  & 0xFF);
  header[idx++] = (uint8_t)((ms >> 16) & 0xFF);
  header[idx++] = (uint8_t)((ms >> 24) & 0xFF);

  header[idx++] = (uint8_t)(int8_t)info->rx_ctrl.rssi;
  header[idx++] = (uint8_t)(int8_t)info->rx_ctrl.noise_floor;

  for (int i = 0; i < 6; i++) header[idx++] = info->mac[i];

  uint16_t csi_len = info->len;
  if (csi_len > CSI_MAX_BYTES) csi_len = CSI_MAX_BYTES;
  header[idx++] = (uint8_t)(csi_len & 0xFF);
  header[idx++] = (uint8_t)((csi_len >> 8) & 0xFF);

  // Emit: magic + header + csi bytes
  Serial.write(0xAA);
  Serial.write(0x55);
  Serial.write(header, sizeof(header));
  Serial.write((const uint8_t*)info->buf, csi_len);
}

void on_csi(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;

  // Debug every 100 frames — remove once confirmed healthy
  static uint32_t debug_count = 0;
  if (debug_count++ % 100 == 0) {
    Serial.printf("# CSI_LEN=%d RSSI=%d NF=%d sig_mode=%d\n",
      info->len,
      info->rx_ctrl.rssi,
      info->rx_ctrl.noise_floor,
      info->rx_ctrl.sig_mode
    );
  }

  emit_csi_frame(info);
}

static void init_csi() {
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&on_csi, NULL));

  wifi_csi_config_t cfg = {};
  cfg.lltf_en           = 1;
  cfg.htltf_en          = 1;
  cfg.stbc_htltf2_en    = 0;
  cfg.ltf_merge_en      = 0;
  cfg.channel_filter_en = 0; // disabled — preserves subcarrier variance
  cfg.manu_scale        = 0;
  cfg.shift             = 0;

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(BAUD_RATE);
  delay(200);

  WiFi.mode(WIFI_AP);

  esp_wifi_set_protocol(WIFI_IF_AP,
    WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);

  bool ok = WiFi.softAP(AP_SSID, AP_PASS, WIFI_CHANNEL, /*hidden=*/false, /*max_conn=*/8);
  if (!ok) {
    Serial.println("AP_START_FAILED");
    return;
  }

  ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

  Serial.println("LISTENER_AP_READY");

  init_csi();
}

void loop() {
  delay(1000);
}
