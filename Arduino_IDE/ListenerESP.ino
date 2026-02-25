#include <Arduino.h>
#include <WiFi.h>

extern "C" {
  #include "esp_wifi.h"
  #include "esp_now.h"
}

// --------- Settings ----------
#define WIFI_CHANNEL 6              // MUST match the sender's channel
#define PRINT_CSI_BYTES 64          // Print only first N CSI bytes to keep serial manageable

// Filter ESP-NOW to one sender
static const bool FILTER_ONE_SENDER = true;
static const uint8_t TARGET_MAC[6]  = {0x68, 0xFE, 0x71, 0x90, 0x60, 0xA0};

// Tie CSI to ESP-NOW by time gating
#define CSI_GATE_MS 30              // window after ESPNOW_RX where we accept CSI

// --------- State ----------
volatile uint32_t last_espnow_ms = 0;
volatile uint32_t last_seq = 0;
volatile bool need_csi = false;

// --------- Utilities ----------
static inline bool mac_eq(const uint8_t *a, const uint8_t *b) {
  for (int i = 0; i < 6; i++) if (a[i] != b[i]) return false;
  return true;
}

static void print_mac(const uint8_t *mac) {
  Serial.printf("%02X:%02X:%02X:%02X:%02X:%02X",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

// --------- ESP-NOW RX callback ----------
void on_espnow_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (!info || !info->src_addr) return;
  if (FILTER_ONE_SENDER && !mac_eq(info->src_addr, TARGET_MAC)) return;

  last_espnow_ms = millis();
  need_csi = true;  // <-- ENABLE CSI logging window

  // Read seq from the packet (first 4 bytes)
  uint32_t seq = 0;
  if (data && len >= 4) memcpy(&seq, data, 4);
  last_seq = seq;

  Serial.print("ESPNOW_RX,");
  Serial.print(last_espnow_ms);
  Serial.print(",seq=");
  Serial.print(seq);
  Serial.print(",");
  print_mac(info->src_addr);
  Serial.print(",len=");
  Serial.println(len);
}

// --------- CSI callback ----------
void on_csi(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;

  uint32_t now = millis();

  if (!need_csi) return;
  if ((now - last_espnow_ms) > CSI_GATE_MS) return;

  int rssi = info->rx_ctrl.rssi;

  Serial.print("CSI,");
  Serial.print(now);
  Serial.print(",seq=");
  Serial.print(last_seq);
  Serial.print(",mac=");
  print_mac(info->mac);
  Serial.print(",rssi=");
  Serial.print(rssi);
  Serial.print(",csi_len=");
  Serial.print(info->len);
  Serial.print(",csi=");

  int n = (info->len < PRINT_CSI_BYTES) ? info->len : PRINT_CSI_BYTES;
  for (int i = 0; i < n; i++) {
    Serial.print((int)info->buf[i]);
    if (i < n - 1) Serial.print(" ");
  }
  Serial.println();
}

static void init_csi() {
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&on_csi, NULL));

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = 1;
  cfg.htltf_en = 1;
  cfg.stbc_htltf2_en = 1;
  cfg.ltf_merge_en = 1;
  cfg.channel_filter_en = 1;   // keep CSI on our channel
  cfg.manu_scale = 0;
  cfg.shift = 0;

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void setup() {
  Serial.begin(912600);
  delay(200);

  WiFi.mode(WIFI_STA);
  delay(100);

  ESP_ERROR_CHECK(esp_wifi_start());
  ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

  wifi_promiscuous_filter_t filt = {};
  filt.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT; // ESP-NOW tends to show here
  ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filt));
  ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }
  ESP_ERROR_CHECK(esp_now_register_recv_cb(on_espnow_recv));

  init_csi();

  Serial.println("LISTENER_READY");
}

void loop() {
  delay(1000);
}