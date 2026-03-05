// This program is used to find the MAC address of the shouter ESP32s

#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA); 
  Serial.println("MAC: " + WiFi.macAddress());
}

void loop() {
  Serial.println("MAC: " + WiFi.macAddress());
  delay(1000);
}