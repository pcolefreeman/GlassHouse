#include <Arduino.h>
#include <WiFi.h>

extern "C" {
  #include "esp_wifi.h"
  #include "esp_now.h"
}

// --------- Settings ----------
#define WIFI_CHANNEL 6
#define SEND_INTERVAL_MS 5

static uint8_t BCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

typedef struct __attribute__((packed)) {
  uint32_t seq;
  uint32_t ms;
  uint8_t  pad[8];
} shout_pkt_t;

static uint32_t seqno = 0;

// NEW callback signature (Arduino-ESP32 core 3.x)
static void on_sent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  // info can be null in some cases; don’t assume it exists
  if ((seqno % 100) == 0) {
    Serial.print("ESPNOW_TX_STATUS,seq=");
    Serial.print(seqno);
    Serial.print(",status=");
    Serial.println(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
  }
}

static void init_espnow_broadcast_peer() {
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BCAST, 6);
  peer.channel = WIFI_CHANNEL;
  peer.encrypt = false;

  esp_err_t e = esp_now_add_peer(&peer);
  if (e != ESP_OK && e != ESP_ERR_ESPNOW_EXIST) {
    Serial.print("Failed to add broadcast peer, err=");
    Serial.println((int)e);
  }
}

void setup() {
  Serial.begin(912600);
  delay(200);

  WiFi.mode(WIFI_STA);
  delay(100);

  ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  // register NEW signature callback
  ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));

  init_espnow_broadcast_peer();

  Serial.print("SHOUTER_READY,mac=");
  Serial.println(WiFi.macAddress());
}

void loop() {
  shout_pkt_t pkt;
  pkt.seq = seqno++;
  pkt.ms  = millis();
  memset(pkt.pad, 0xA5, sizeof(pkt.pad));

  esp_err_t err = esp_now_send(BCAST, (uint8_t*)&pkt, sizeof(pkt));
  if (err != ESP_OK) {
    Serial.print("ESPNOW_TX_ERR,code=");
    Serial.println((int)err);
  } else if ((pkt.seq % 100) == 0) {
    Serial.print("ESPNOW_TX,seq=");
    Serial.print(pkt.seq);
    Serial.print(",ms=");
    Serial.println(pkt.ms);
  }

  delay(SEND_INTERVAL_MS);
}