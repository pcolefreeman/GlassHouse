#include <Arduino.h>
#include <WiFi.h>
#include "esp_wifi.h"

#define WIFI_CHANNEL 1
const uint8_t TARGET_MAC[] = {0x68, 0xFE, 0x71, 0x90, 0x60, 0xA0};

// The callback stays the same
void handle_csi_data(void *ctx, wifi_csi_info_t *data) {
    if (!data || !data->buf){
            Serial.println("Invalid CSI data received.");
            return;
    }

    if (memcmp(data->mac, TARGET_MAC, 6) != 0) { //comment out to recieve all data, not just from the specific Node
        return; // This packet is not from our specific Node, ignore it.
    } 

    Serial.print("BRAIN_DATA,");
    
    // for (int i = 0; i < 6; i++) { // Print MAC address
    //     Serial.printf("%02x%s", data->mac[i], (i < 5) ? ":" : "");
    // }
    
    //Serial.printf(",%d,", data->rx_ctrl.rssi);

    // Print RSSI for signal strength tracking
    Serial.printf("RSSI:%d,CSI_LEN:%d,DATA:", data->rx_ctrl.rssi, data->len);

    /*
    Serial.print("BRAIN_DATA,");
    
    char macStr[18];
    snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
             data->mac[0], data->mac[1], data->mac[2], 
             data->mac[3], data->mac[4], data->mac[5]);
    Serial.print(macStr);
    
    Serial.print(",");
    Serial.print(data->rx_ctrl.rssi);
    Serial.print(",");
    */

    int8_t *my_buf = (int8_t *)data->buf;
    for (int i = 0; i < data->len; i++) {
        Serial.print(my_buf[i]);
        if (i < data->len - 1) Serial.print(" ");
    }
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    
    WiFi.mode(WIFI_STA);
    esp_wifi_stop(); // Reset wifi to ensure clean config
    esp_wifi_start();

    // Force same WiFi channel
    esp_wifi_set_promiscuous(true);

    wifi_promiscuous_filter_t filter = {.filter_mask = WIFI_PROMIS_FILTER_MASK_DATA};
    esp_wifi_set_promiscuous_filter(&filter);

    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

    // Proper CSI configuration
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
}

void loop() {
    delay(1000); // dummy loop to keep the program running and allow CSI data reception
}