#include <esp_now.h>
#include <WiFi.h>
#include "esp_wifi.h"
#include <string.h>

#define WIFI_CHANNEL 1

void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Force channel
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }

    esp_now_peer_info_t peerInfo = {};
    uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add peer");
        return;
    }

    Serial.printf("Starting Node %d on Channel %d\n", NODE_ID, WIFI_CHANNEL);
    delay(NODE_ID * 330); 
}

void loop() {
    uint8_t data = NODE_ID;
    uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    
    esp_err_t result = esp_now_send(broadcastAddress, &data, sizeof(data));
    
    if (result == ESP_OK) {
        Serial.printf("Node %d: Broadcast sent successfully\n", NODE_ID);
    } else {
        Serial.println("Send failed");
    }
    
    Serial.println(WiFi.macAddress());

    delay(2000); 
}