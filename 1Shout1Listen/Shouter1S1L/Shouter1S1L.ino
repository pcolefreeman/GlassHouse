// ShouterAP_UDP.ino
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

extern "C" {
  #include "esp_wifi.h"
}

// -------------------- CONFIG --------------------
#define BAUD_RATE        921600
#define WIFI_CHANNEL     6
#define AP_SSID          "CSI_PRIVATE_AP"
#define AP_PASS          "12345678"

#define LISTENER_IP      "192.168.4.1"
#define UDP_PORT         3333
#define SEND_INTERVAL_MS 500

#define SHOUTER_ID       1   // *** change to 2, 3, 4 on each respective shouter
#define NUM_SHOUTERS      3   // *** change based on number of shouters currently in the system

// -------------------- UDP --------------------
WiFiUDP Udp;
IPAddress listenerIP;

typedef struct __attribute__((packed)) {
  uint32_t seq;
  uint32_t ms;
  uint8_t  shouter_id;
  uint8_t  pad[99];   // large payload forces longer HT frame → better CSI
} shout_pkt_t;

static uint32_t seqno = 0;

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(BAUD_RATE);
  delay(200);

  // Must set protocol BEFORE WiFi.mode() — not just before WiFi.begin()
  esp_err_t err = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  if (err != ESP_OK) {
    Serial.printf("set_protocol failed: %d\n", err);
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  // Explicitly set 802.11n HT20 before connecting
  esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);

  Serial.print("Connecting to AP ");
  Serial.println(AP_SSID);

  WiFi.begin(AP_SSID, AP_PASS);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
    if (millis() - start > 15000) {
      Serial.println("\nAP_CONNECT_TIMEOUT");
      return;
    }
  }
  Serial.println("\nAP_CONNECTED");

  Serial.print("STA_MAC=");
  Serial.println(WiFi.macAddress());
  Serial.print("STA_IP=");
  Serial.println(WiFi.localIP());

  // Force MCS0 LGI — most conservative HT rate, most likely to stick
  err = esp_wifi_config_80211_tx_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS0_LGI);
  if (err != ESP_OK) {
    Serial.printf("set_tx_rate failed: %d\n", err);
  } else {
    Serial.println("TX_RATE=MCS0_LGI set");
  }

  listenerIP.fromString(LISTENER_IP);
  Udp.begin(0);

  Serial.print("UDP_TARGET=");
  Serial.print(listenerIP);
  Serial.print(":");
  Serial.println(UDP_PORT);

  // Stagger transmission start by node ID to avoid collisions
  uint32_t offset_ms = (SHOUTER_ID - 1) * (SEND_INTERVAL_MS / NUM_SHOUTERS);  
  Serial.printf("TX_OFFSET=%ums\n", offset_ms);
  delay(offset_ms);

  Serial.println("SHOUTER_READY");
}

// -------------------- LOOP --------------------
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WIFI_LOST — reconnecting...");
    WiFi.reconnect();
    delay(1000);
    return;
  }

  shout_pkt_t pkt;
  pkt.seq        = seqno++;
  pkt.ms         = millis();
  pkt.shouter_id = SHOUTER_ID;
  memset(pkt.pad, 0xA5, sizeof(pkt.pad));

  Udp.beginPacket(listenerIP, UDP_PORT);
  Udp.write((uint8_t*)&pkt, sizeof(pkt));
  bool ok = Udp.endPacket();

  if ((pkt.seq % 200) == 0) {
    Serial.print("UDP_TX,seq=");
    Serial.print(pkt.seq);
    Serial.print(",ms=");
    Serial.print(pkt.ms);
    Serial.print(",ok=");
    Serial.println(ok ? "1" : "0");
  }

  delay(SEND_INTERVAL_MS);
}