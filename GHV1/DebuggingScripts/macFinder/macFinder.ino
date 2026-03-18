// This program is used to find the MAC address of the listener and shouter ESP32s

#include <string.h>
#include <WiFi.h>
#include <map>

std::map<String, int> mac_dict{
  {"68:FE:71:90:66:A8" , 84}, // listener ID
  {"68:FE:71:90:60:A0", 1},
  {"68:FE:71:90:68:14", 2},
  {"68:FE:71:90:6B:90", 3},
  {"20:E7:C8:EC:F5:DC": 4}
};

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA); 

  Serial.println("MAC: " + WiFi.macAddress());
  int id = mac_dict[WiFi.macAddress()];
}

void loop() {
  Serial.println("MAC: " + WiFi.macAddress());
  Serial.printf("This is Node %d\n", mac_dict[WiFi.macAddress()]);
  delay(1000);
}