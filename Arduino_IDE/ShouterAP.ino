// ShouterAP_UDP.ino
// ESP32 connects to the Listener's PRIVATE SoftAP and repeatedly sends UDP packets.
// Baud stays 912600.

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

extern "C" {
  #include "esp_wifi.h"
}

// -------------------- CONFIG --------------------
#define BAUD_RATE        912600
#define WIFI_CHANNEL     6                 // should match AP channel (usually auto when connecting)
#define AP_SSID          "CSI_PRIVATE_AP"
#define AP_PASS          "12345678"

#define LISTENER_IP      "192.168.4.1"     // default SoftAP IP
#define UDP_PORT         3333
#define SEND_INTERVAL_MS 5                 // start 10–20ms if serial/logging is heavy

// -------------------- UDP --------------------
WiFiUDP Udp;
IPAddress listenerIP;

typedef struct __attribute__((packed)) {
  uint32_t seq;
  uint32_t ms;
  uint8_t  pad[8];
} shout_pkt_t;

static uint32_t seqno = 0;

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(BAUD_RATE);
  delay(200);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // reduce latency/jitter

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

  // Optional: force channel (usually not needed; association sets it)
  ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

  listenerIP.fromString(LISTENER_IP);

  // Bind UDP (optional, but fine)
  Udp.begin(0);

  Serial.print("UDP_TARGET=");
  Serial.print(listenerIP);
  Serial.print(":");
  Serial.println(UDP_PORT);

  Serial.println("SHOUTER_READY");
}

// -------------------- LOOP --------------------
void loop() {
  shout_pkt_t pkt;
  pkt.seq = seqno++;
  pkt.ms  = millis();
  memset(pkt.pad, 0xA5, sizeof(pkt.pad));

  Udp.beginPacket(listenerIP, UDP_PORT);
  Udp.write((uint8_t*)&pkt, sizeof(pkt));
  bool ok = Udp.endPacket();

  // Light heartbeat
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