#include <WiFi.h>
#include "esp_wifi.h"
#include <string.h>

#define WIFI_CHANNEL 1

// Optional: filter to one sender MAC. Comment out to accept all.
const uint8_t TARGET_MAC[] = {0x68,0xFE,0x71,0x90,0x60,0xA0};
#define FILTER_ONE_MAC 1   // set to 0 to accept all

static void printMac(const uint8_t mac[6]) {
  Serial.printf("%02x:%02x:%02x:%02x:%02x:%02x",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void handle_csi_data(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len <= 0) return;

#if FILTER_ONE_MAC
  if (memcmp(data->mac, TARGET_MAC, 6) != 0) return;
#endif

  // CSV: tag, timestamp, mac, rssi, csi_len, then CSI samples as ints
  // Serial.print("BRAIN_DATA,");
  // Serial.print(millis());
  // Serial.print(",");

  // printMac(data->mac);
  // Serial.print(",");

  // Serial.print((int)data->rx_ctrl.rssi);
  // Serial.print(",");

  // Serial.print((int)data->len);

  // int8_t *csi = (int8_t *)data->buf;
  // for (int i = 0; i < data->len; i++) {
  //   Serial.print(",");
  //   Serial.print((int)csi[i]);
  // }
  // Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(200);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true, true);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);

  esp_wifi_stop();
  esp_wifi_start();

  // Force channel
  esp_wifi_set_promiscuous(true);
  wifi_promiscuous_filter_t filter = {
    .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA
    // if needed: WIFI_PROMIS_FILTER_MASK_ALL
  };
  esp_wifi_set_promiscuous_filter(&filter);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  uint8_t ch; wifi_second_chan_t sc;
  esp_wifi_get_channel(&ch, &sc);
  Serial.printf("BRAIN CHANNEL: %u\n", ch);
  Serial.print("BRAIN MAC: "); Serial.println(WiFi.macAddress());

  // CSI config (keep what you had)
  wifi_csi_config_t config = {
    .lltf_en = true,
    .htltf_en = true,
    .stbc_htltf2_en = true,
    .ltf_merge_en = true,
    .channel_filter_en = true,
    .manu_scale = false,
    .shift = false
  };

  esp_wifi_set_csi_config(&config);
  esp_wifi_set_csi_rx_cb(handle_csi_data, NULL);
  esp_wifi_set_csi(true);

  Serial.println("Brain Unit Ready.");
  Serial.println(WiFi.macAddress());//deubgger

}

void loop() {
  delay(1000);
}
