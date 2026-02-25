#include <esp_now.h>
#include <WiFi.h>
#include "esp_wifi.h"
#include <string.h>

#define WIFI_CHANNEL 1
#define NODE_ID 1   // CHANGE PER NODE

static uint8_t broadcastAddress[] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

typedef struct __attribute__((packed)) {
  uint8_t node_id;
  uint32_t seq;
  uint32_t ms;
} payload_t;

uint32_t seqNum = 0;

// NEW callback signature (ESP32 Arduino core v3.x)
void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  Serial.print("ESP-NOW send: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
}

static void forceWiFiChannel(uint8_t channel) {
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  WiFi.mode(WIFI_S
TA);
  WiFi.disconnect(true, true);
  delay(100);

  forceWiFiChannel(WIFI_CHANNEL);

  uint8_t ch; wifi_second_chan_t sc;
  esp_wifi_get_channel(&ch, &sc);
  Serial.printf("SENDER CHANNEL: %u\n", ch);
  Serial.print("SENDER MAC: "); Serial.println(WiFi.macAddress());


  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_send_cb(OnDataSent);

  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = WIFI_CHANNEL;
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add broadcast peer");
    return;
  }

  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.printf("Starting Node %d on Channel %d\n", NODE_ID, WIFI_CHANNEL);

  delay(NODE_ID * 330);
}

void loop() {
  payload_t p;
  p.node_id = NODE_ID;
  p.seq = seqNum++;
  p.ms  = millis();

  esp_err_t result = esp_now_send(broadcastAddress, (uint8_t*)&p, sizeof(p));

  if (result == ESP_OK) {
    Serial.printf("Node %d: queued (seq=%lu)\n", NODE_ID, (unsigned long)p.seq);
  } else {
    Serial.printf("Send failed: %d\n", (int)result);
  }

  Serial.println(WiFi.macAddress());//deubgger

  delay(2000);
}