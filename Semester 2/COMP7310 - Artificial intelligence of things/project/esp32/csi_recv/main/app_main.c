/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/


/**
 * In this file, the following code blocks are marked for customization.
 * Each block starts with the comment: "// YOUR CODE HERE" 
 * and ends with: "// END OF YOUR CODE".
 *
 * [1] Modify the CSI Buffer and FIFO Lengths:
 *     - Adjust the buffer configuration based on your system if necessary.
 *
 * [2] Implement Algorithms:
 *     - Develop algorithms for motion detection, breathing rate estimation, and MQTT message sending.
 *     - Implement them in their respective functions.
 *
 * [3] Modify Wi-Fi Configuration:
 *     - Modify the Wi-Fi settings–SSID and password to connect to your router.
 *
 * [4] Finish the function `csi_process()`:
 *     - Fill in the group information.
 *     - Process and analyze CSI data in the `csi_process` function.
 *     - Implement your algorithms in this function if on-board. (Task 2)
 *     - Return the results to the host or send the CSI data via MQTT. (Task 3)
 *
 * Feel free to modify these sections to suit your project requirements!
 * 
 * Have fun building!
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"

#include "mqtt_client.h"
#include "esp_transport.h"
#include "esp_transport_tcp.h"
#include "esp_timer.h"

#include "esp_dsp.h"

#include <math.h>
#include <float.h>


// [1] YOUR CODE HERE
#define CSI_BUFFER_LENGTH 800
#define CSI_FIFO_LENGTH 100
static int16_t CSI_Q[CSI_BUFFER_LENGTH];
static int CSI_Q_INDEX = 0; // CSI Buffer Index
// Enable/Disable CSI Buffering. 1: Enable, using buffer, 0: Disable, using serial output
static bool CSI_Q_ENABLE = 1; 
static void csi_process(const int8_t *csi_data, int length);

#define RSSI_BUFFER_LENGTH 50  // RSSI历史数据长度
static int8_t RSSI_BUFFER[RSSI_BUFFER_LENGTH]; // 存储RSSI历史数据
static int RSSI_INDEX = 0; // RSSI缓冲区索引
static int8_t current_rssi = 0; // 当前RSSI值

// MQTT Configuration
#define MQTT_BROKER_URL "mqtt://192.168.10.28"
#define MQTT_TOPIC "flask/esp32/bjh"
#define MQTT_PORT 1883
static const char *TAG;
static esp_mqtt_client_handle_t mqtt_client = NULL;
static bool mqtt_connected = false;

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%d", base, event_id);
    esp_mqtt_event_handle_t event = event_data;
    
    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
        mqtt_connected = true;
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
        mqtt_connected = false;
        break;
    case MQTT_EVENT_PUBLISHED:
        ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
        break;
    default:
        ESP_LOGI(TAG, "Other MQTT event id:%d", event->event_id);
        break;
    }
}

static void mqtt_init(void) {
    ESP_LOGI(TAG, "Initializing MQTT client...");
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URL,
        .broker.address.port = MQTT_PORT,
        .network.timeout_ms = 5000,
        .network.reconnect_timeout_ms = 5000,
        .session.keepalive = 300, // 5 minutes
        
    };

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (mqtt_client == NULL) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return;
    }
    
    // Register event handler
    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(mqtt_client);
    ESP_LOGI(TAG, "MQTT client initialized");
}

// [1] END OF YOUR CODE


// [2] YOUR CODE HERE
// Modify the following functions to implement your algorithms.
// NOTE: Please do not change the function names and return types.
bool motion_detection(int* flag) {
    // 检查是否有足够的RSSI样本
    if (RSSI_INDEX < 5) {
        ESP_LOGW(TAG, "Not enough RSSI data: %d", RSSI_INDEX);
        *flag = 0;
        return false;
    }

    *flag = 1; //

    // log rssi
    char rssi_log[256];
    snprintf(rssi_log, sizeof(rssi_log), "RSSI Buffer: ");
    for (int i = 0; i < RSSI_INDEX; i++) {
        snprintf(rssi_log + strlen(rssi_log), sizeof(rssi_log) - strlen(rssi_log), "%d ", RSSI_BUFFER[i]);
    }
    ESP_LOGI(TAG, "%s", rssi_log);

    // 计算RSSI的变动值
    double sum = 0.0, mean = 0.0, variance = 0.0, stddev = 0.0;
    int8_t min_rssi = INT8_MAX;
    int8_t max_rssi = INT8_MIN;

    // 计算平均值和寻找最值
    for (int i = 0; i < RSSI_INDEX; i++) {
        sum += RSSI_BUFFER[i];
        if (RSSI_BUFFER[i] < min_rssi) min_rssi = RSSI_BUFFER[i];
        if (RSSI_BUFFER[i] > max_rssi) max_rssi = RSSI_BUFFER[i];
    }
    mean = sum / RSSI_INDEX;

    // 计算方差
    for (int i = 0; i < RSSI_INDEX; i++) {
        variance += (RSSI_BUFFER[i] - mean) * (RSSI_BUFFER[i] - mean);
    }
    variance /= RSSI_INDEX;

    // 计算标准差
    stddev = sqrt(variance);
    
    // 输出RSSI统计信息
    ESP_LOGI(TAG, "RSSI Stats - Current: %d dBm, Min: %d dBm, Max: %d dBm, Mean: %.2f dBm, StdDev: %.2f dBm",
             current_rssi, min_rssi, max_rssi, mean, stddev);
    
    
    // 如果RSSI变化超过阈值，判断为有运动
    // 可以根据实际情况调整阈值
    const double RSSI_STDDEV_THRESHOLD = 1.0;  // RSSI标准差阈值

    bool motion_detected = (stddev > RSSI_STDDEV_THRESHOLD);
    
    ESP_LOGI(TAG, "Motion Detection Result (RSSI-based): %s", 
             motion_detected ? "MOTION DETECTED" : "NO MOTION");
    
    return motion_detected;
}

#define BREATH_SAMPLE_RATE_HZ 100
#define BREATH_MIN_PERIOD 250 // 60s / 24BPM * 100Hz
#define BREATH_MAX_PERIOD 750 // 60s / 8BPM * 100Hz
#define LOW_PASS_CUTOFF 0.5f
#define HIGH_PASS_CUTOFF 0.1f

static float csi_f32[CSI_BUFFER_LENGTH];
static float filtered[CSI_BUFFER_LENGTH];
static float corr[2 * CSI_BUFFER_LENGTH - 1];
static float biquad_state[4]; // State buffer for biquad filter

#include "esp_task_wdt.h"

void compute_autocorrelation_debug(const float *x, int len, float *result) {
    for (int i = 0; i < 2 * len - 1; i++) {
        result[i] = 0.0f;
    }

    int mid = len - 1;
    ESP_LOGI(TAG, "Computing autocorrelation, signal length: %d, mid point: %d", len, mid);

    for (int lag = 0; lag < len; lag++) {
        float sum = 0.0f;
        for (int i = 0; i < len - lag; i++) {
            sum += x[i] * x[i + lag];
        }
        result[mid + lag] = sum;


        if (lag < 5 || lag > len - 5) {
            ESP_LOGD(TAG, "Autocorr[lag=%d] = %.5f", lag, sum);
        }
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); // Yield to other tasks

    for (int lag = 1; lag < len; lag++) {
        float sum = 0.0f;
        for (int i = 0; i < len - lag; i++) {
            sum += x[i + lag] * x[i];
        }
        result[mid - lag] = sum;
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); // Yield to other tasks
    
    ESP_LOGI(TAG, "Autocorrelation computation complete");
}



int breathing_rate_estimation(int* flag) {
    if (CSI_Q_INDEX < CSI_BUFFER_LENGTH) {
        ESP_LOGW(TAG, "Not enough CSI data: %d/%d", CSI_Q_INDEX, CSI_BUFFER_LENGTH);
        *flag = 0;
        return 0;
    }

    *flag = 1;

    // 先检查CSI数据是否有效变化
    bool has_variation = false;
    int16_t first_val = CSI_Q[0];
    for (int i = 1; i < 20; i++) { // 检查前20个样本是否有变化
        if (abs(CSI_Q[i] - first_val) > 5) { // 5是差异阈值
            has_variation = true;
            break;
        }
    }

    if (!has_variation) {
        ESP_LOGW(TAG, "CSI data has insufficient variation in the first 20 samples");
    }

    // 计算CSI数据的统计信息
    int16_t min_val = INT16_MAX, max_val = INT16_MIN;
    double sum = 0.0;
    for (int i = 0; i < CSI_BUFFER_LENGTH; i++) {
        if (CSI_Q[i] < min_val) min_val = CSI_Q[i];
        if (CSI_Q[i] > max_val) max_val = CSI_Q[i];
        sum += CSI_Q[i];
    }
    double mean = sum / CSI_BUFFER_LENGTH;
    
    ESP_LOGI(TAG, "==== CSI_Q Statistics ====");
    ESP_LOGI(TAG, "Min: %d, Max: %d, Mean: %.2f, Range: %d", 
             min_val, max_val, mean, max_val - min_val);
    
    ESP_LOGI(TAG, "==== CSI_Q Raw Sample (first 10 values) ====");
    for (int i = 0; i < 10; i++) {
        // ESP_LOGI(TAG, "CSI_Q[%d] = %d", i, CSI_Q[i]);
    }

    // Step 1: int16 -> float并去除直流分量
    for (int i = 0; i < CSI_BUFFER_LENGTH; i++) {
        csi_f32[i] = (float)(CSI_Q[i] - mean); // 减去平均值去除直流分量
    }

    ESP_LOGI(TAG, "==== CSI_F32 Float Sample (DC removed) ====");
    float dc_removed_min = FLT_MAX, dc_removed_max = -FLT_MAX;
    for (int i = 0; i < CSI_BUFFER_LENGTH; i++) {
        if (csi_f32[i] < dc_removed_min) dc_removed_min = csi_f32[i];
        if (csi_f32[i] > dc_removed_max) dc_removed_max = csi_f32[i];
    }
    ESP_LOGI(TAG, "DC removed range: %.2f to %.2f", dc_removed_min, dc_removed_max);
    
    for (int i = 0; i < 10; i++) {
        // ESP_LOGI(TAG, "csi_f32[%d] = %.5f", i, csi_f32[i]);
    }

    // Step 2: 应用带通滤波器 (LPF + HPF)
    float coeffs[5];
    memset(biquad_state, 0, sizeof(biquad_state));
    
    // 低通滤波 - 截止频率设为0.3Hz (18 BPM)
    dsps_biquad_gen_lpf_f32(coeffs, LOW_PASS_CUTOFF / (BREATH_SAMPLE_RATE_HZ / 2.0f), 0.707f);
    dsps_biquad_f32_ansi(csi_f32, filtered, CSI_BUFFER_LENGTH, coeffs, biquad_state);
    ESP_LOGI(TAG, "Low-pass filter applied with cutoff: %.2f Hz", 
             LOW_PASS_CUTOFF);

    // 高通滤波 - 截止频率设为0.05Hz (3 BPM)
    memset(biquad_state, 0, sizeof(biquad_state));
    dsps_biquad_gen_hpf_f32(coeffs, HIGH_PASS_CUTOFF / (BREATH_SAMPLE_RATE_HZ / 2.0f), 0.707f);
    dsps_biquad_f32_ansi(filtered, csi_f32, CSI_BUFFER_LENGTH, coeffs, biquad_state);
    ESP_LOGI(TAG, "High-pass filter applied with cutoff: %.2f Hz", 
             HIGH_PASS_CUTOFF);

    // 检查滤波后的信号统计信息
    float filtered_min = FLT_MAX, filtered_max = -FLT_MAX;
    float filtered_sum = 0.0f, filtered_abs_sum = 0.0f;
    for (int i = 0; i < CSI_BUFFER_LENGTH; i++) {
        if (csi_f32[i] < filtered_min) filtered_min = csi_f32[i];
        if (csi_f32[i] > filtered_max) filtered_max = csi_f32[i];
        filtered_sum += csi_f32[i];
        filtered_abs_sum += fabsf(csi_f32[i]);
    }
    float filtered_mean = filtered_sum / CSI_BUFFER_LENGTH;
    float filtered_abs_mean = filtered_abs_sum / CSI_BUFFER_LENGTH;
    
    ESP_LOGI(TAG, "==== Filtered CSI Statistics ====");
    ESP_LOGI(TAG, "Min: %.5f, Max: %.5f, Mean: %.5f, Abs Mean: %.5f", 
             filtered_min, filtered_max, filtered_mean, filtered_abs_mean);

    ESP_LOGI(TAG, "==== Filtered CSI (first 10 values) ====");
    for (int i = 0; i < 10; i++) {
        // ESP_LOGI(TAG, "Filtered[%d] = %.5f", i, csi_f32[i]);
    }

    // 如果滤波后的信号太弱，提前返回
    if (filtered_max - filtered_min < 0.1f) {
        ESP_LOGW(TAG, "Filtered signal too weak (range: %.5f)", filtered_max - filtered_min);
        return 0;
    }

    // Step 3: 计算自相关
    ESP_LOGI(TAG, "Computing autocorrelation...");
    compute_autocorrelation_debug(csi_f32, CSI_BUFFER_LENGTH, corr);

    // Step 4: 归一化并寻找周期
    float energy = 0;
    for (int i = 0; i < CSI_BUFFER_LENGTH; i++) {
        energy += csi_f32[i] * csi_f32[i];
    }
    if (energy < 1e-6f) {
        ESP_LOGW(TAG, "Signal energy too low: %.8f", energy);
        return 0;
    }

    int max_idx = 0;
    float max_corr_val = -1e10;
    int mid = CSI_BUFFER_LENGTH - 1;

    ESP_LOGI(TAG, "==== Searching for breathing period (lag=%d~%d) ====", 
             BREATH_MIN_PERIOD, BREATH_MAX_PERIOD);
    
    // 在呼吸率的有效周期范围内搜索自相关峰值
    for (int i = mid + BREATH_MIN_PERIOD; i <= mid + BREATH_MAX_PERIOD && i < (2 * CSI_BUFFER_LENGTH - 1); i++) {
        float val = corr[i];
        if (energy > 1e-6f) {
            val /= energy;  // 归一化
        }
        
        // int lag = i - mid;
        // ESP_LOGI(TAG, "Lag = %d, Corr = %.5f", lag, val);
        
        if (val > max_corr_val) {
            max_corr_val = val;
            max_idx = i;
        }
    }

    // 验证检测到的峰值
    int period_samples = max_idx - mid;
    if (period_samples <= 0) {
        ESP_LOGW(TAG, "Invalid period_samples: %d", period_samples);
        return 0;
    }

    // 降低阈值以适应可能的弱信号
    // if (max_corr_val < 0.01f) {  // 降低阈值到0.01
    //     ESP_LOGW(TAG, "Peak correlation too low: %.5f (< 0.01)", max_corr_val);
    //     return 0;
    // }

    // 计算呼吸率
    float period_sec = (float)period_samples / BREATH_SAMPLE_RATE_HZ;
    int bpm = (int)(60.0f / period_sec);

    // 呼吸率合理性检查
    if (bpm < 8 || bpm > 24) {
        ESP_LOGW(TAG, "Calculated BPM outside normal range: %d BPM (expected 8-24)", bpm);
        return 0;
    }

    // ESP_LOGI(TAG, "Detected Peak Lag = %d samples (%.2f seconds)", period_samples, period_sec);
    // ESP_LOGI(TAG, "Correlation value at peak: %.5f", max_val);
    // ESP_LOGI(TAG, "Estimated Breathing Rate: %d BPM", bpm);

    return bpm;
}

void mqtt_send(bool motion, int breathing_rate) {
    static int64_t last_publish_time = 0;
    int64_t current_time = esp_timer_get_time() / 1000; // Convert to milliseconds
    
    // Only publish every 2 seconds
    if (current_time - last_publish_time < 2000) {
        ESP_LOGD(TAG, "Skipping MQTT publish, last publish was %lld ms ago", 
                 current_time - last_publish_time);
        return;
    }
    
    // Format the JSON payload
    char mqtt_payload[100];
    snprintf(mqtt_payload, sizeof(mqtt_payload), 
             "{\"motion\":%s, \"breathing_rate\":%d}", 
             motion ? "true" : "false", breathing_rate);
    
    ESP_LOGI(TAG, "MQTT sending: %s", mqtt_payload);
    
    // Publish the message if client is initialized and connected
    if (mqtt_client != NULL && mqtt_connected) {
        int msg_id = esp_mqtt_client_publish(mqtt_client, 
                                             MQTT_TOPIC, 
                                             mqtt_payload, 
                                             0, // Message length (0 = calculate from string)
                                             0, // QoS level 0
                                             0  // Not retained
                                             );
        if (msg_id >= 0) {
            ESP_LOGI(TAG, "MQTT message sent successfully, msg_id=%d", msg_id);
            last_publish_time = current_time;
        } else {
            ESP_LOGE(TAG, "Failed to send MQTT message");
        }
    } else {
        ESP_LOGE(TAG, "MQTT client not initialized or not connected");
    }
}

// [2] END OF YOUR CODE


#define CONFIG_LESS_INTERFERENCE_CHANNEL   1
#define CONFIG_WIFI_BAND_MODE   WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT20
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT20
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT20
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   1
#define CONFIG_GAIN_CONTROL                 CONFIG_FORCE_GAIN

// UPDATE: Define parameters for scan method
#if CONFIG_EXAMPLE_WIFI_ALL_CHANNEL_SCAN
#define DEFAULT_SCAN_METHOD WIFI_ALL_CHANNEL_SCAN
#elif CONFIG_EXAMPLE_WIFI_FAST_SCAN
#define DEFAULT_SCAN_METHOD WIFI_FAST_SCAN
#else
#define DEFAULT_SCAN_METHOD WIFI_FAST_SCAN
#endif /*CONFIG_EXAMPLE_SCAN_METHOD*/
//

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};


static const char *TAG = "csi_recv";
typedef struct
{
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 16; /**< reserved */
    unsigned fft_gain : 8;
    unsigned agc_gain : 8;
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
} wifi_pkt_rx_ctrl_phy_t;

#if CONFIG_FORCE_GAIN
    /**
     * @brief Enable/disable automatic fft gain control and set its value
     * @param[in] force_en true to disable automatic fft gain control
     * @param[in] force_value forced fft gain value
     */
    extern void phy_fft_scale_force(bool force_en, uint8_t force_value);

    /**
     * @brief Enable/disable automatic gain control and set its value
     * @param[in] force_en true to disable automatic gain control
     * @param[in] force_value forced gain value
     */
    extern void phy_force_rx_gain(int force_en, int force_value);
#endif

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                             int32_t event_id, void* event_data);
static bool wifi_connected = false;

//------------------------------------------------------WiFi Initialize------------------------------------------------------
static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                      ESP_EVENT_ANY_ID,
                                                      &wifi_event_handler,
                                                      NULL,
                                                      &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                      IP_EVENT_STA_GOT_IP,
                                                      &wifi_event_handler,
                                                      NULL,
                                                      &instance_got_ip));
    
    // [3] YOUR CODE HERE
    // You need to modify the ssid and password to match your Wi-Fi network.
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "Pejoy-2.4G",         
            .password = "bjh123456",
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            // UPDATES: only use this scan method when you want to connect your mobile phone's hotpot
            .scan_method = DEFAULT_SCAN_METHOD,
            //
        
            .pmf_cfg = {
                .capable = true,
                .required = false
            },
        },
    };
    // [3] END OF YOUR CODE

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "wifi_init finished.");
}

//------------------------------------------------------WiFi Event Handler------------------------------------------------------
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                             int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Trying to connect to AP...");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Connection failed! Retrying...");
        wifi_connected = false;
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP:" IPSTR, IP2STR(&event->ip_info.ip));
        wifi_connected = true;
        
        wifi_ap_record_t ap_info;
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            ESP_LOGI(TAG, "Connected to AP - SSID: %s, Channel: %d, RSSI: %d",
                    ap_info.ssid, ap_info.primary, ap_info.rssi);
        }
    }
}

//------------------------------------------------------ESP-NOW Initialize------------------------------------------------------
static void wifi_esp_now_init(esp_now_peer_info_t peer) 
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE, 
        .rate = CONFIG_ESP_NOW_RATE,//  WIFI_PHY_RATE_MCS0_LGI,    
        .ersu = false,                     
        .dcm = false                       
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr,&rate_config));
    ESP_LOGI(TAG, "================ ESP NOW Ready ================");
    ESP_LOGI(TAG, "esp_now_init finished.");
}

//------------------------------------------------------CSI Callback------------------------------------------------------
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;

    // 删除此处日志，减少噪声
    // ESP_LOGI(TAG, "CSI callback triggered");

    // Applying the CSI_Q_ENABLE flag to determine the output method
    if (!CSI_Q_ENABLE) {
        ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d\n",
                   info->len, MAC2STR(info->mac), info->rx_ctrl.rssi,
                   info->rx_ctrl.rate, info->rx_ctrl.noise_floor,
                   info->rx_ctrl.channel);
    } else {
        csi_process(info->buf, info->len);
    }

    // 保存当前RSSI值
    current_rssi = info->rx_ctrl.rssi;
    
    // 将RSSI值添加到缓冲区（不输出日志）
    if (RSSI_INDEX < RSSI_BUFFER_LENGTH) {
        RSSI_BUFFER[RSSI_INDEX++] = current_rssi;
    } else {
        memmove(RSSI_BUFFER, RSSI_BUFFER + 1, (RSSI_BUFFER_LENGTH - 1) * sizeof(int8_t));
        RSSI_BUFFER[RSSI_BUFFER_LENGTH - 1] = current_rssi;
    }

    // 删除此处日志，减少噪声
    // ESP_LOGI(TAG, "CSI callback triggered, RSSI: %d dBm", current_rssi);
    
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    // 只检查MAC地址，不输出日志
    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        // ESP_LOGI(TAG, "MAC address doesn't match, skipping packet");
        return;
    }

    wifi_pkt_rx_ctrl_phy_t *phy_info = (wifi_pkt_rx_ctrl_phy_t *)info;
    static int s_count = 0;

#if CONFIG_GAIN_CONTROL
    static uint16_t agc_gain_sum=0; 
    static uint16_t fft_gain_sum=0; 
    static uint8_t agc_gain_force_value=0; 
    static uint8_t fft_gain_force_value=0; 
    if (s_count<100) {
        agc_gain_sum += phy_info->agc_gain;
        fft_gain_sum += phy_info->fft_gain;
    }else if (s_count == 100) {
        agc_gain_force_value = agc_gain_sum/100;
        fft_gain_force_value = fft_gain_sum/100;
    #if CONFIG_FORCE_GAIN
        phy_fft_scale_force(1,fft_gain_force_value);
        phy_force_rx_gain(1,agc_gain_force_value);
    #endif
        ESP_LOGI(TAG,"fft_force %d, agc_force %d",fft_gain_force_value,agc_gain_force_value);
    }
#endif

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    if (CSI_Q_ENABLE == 0) {
        ESP_LOGI(TAG, "================ CSI RECV via Serial Port ================");
        ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d",
            s_count++, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate,
            rx_ctrl->noise_floor, phy_info->fft_gain, phy_info->agc_gain, rx_ctrl->channel,
            rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->rx_state);
        ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, info->buf[0]);

        for (int i = 1; i < info->len; i++) {
            ets_printf(",%d", info->buf[i]);
        }
        ets_printf("]\"\n");
    }
    // 删除此处日志，减少噪声
    // else {
    //     ESP_LOGI(TAG, "================ CSI RECV via Buffer ================");
    //     csi_process(info->buf, info->len);
    // }
}

//------------------------------------------------------CSI Processing & Algorithms------------------------------------------------------
static void csi_process(const int8_t *csi_data, int length)
{   
    static int64_t last_process_time = 0;
    int64_t current_time = esp_timer_get_time() / 1000; // 转换为毫秒
    bool should_process = (current_time - last_process_time >= 2000);
    
    // 无论是否要处理算法，都继续收集CSI数据（但不输出日志）
    if (CSI_Q_INDEX + length > CSI_BUFFER_LENGTH) {
        int shift_size = CSI_BUFFER_LENGTH - CSI_FIFO_LENGTH;
        memmove(CSI_Q, CSI_Q + CSI_FIFO_LENGTH, shift_size * sizeof(int16_t));
        CSI_Q_INDEX = shift_size;
    }    
    
    // 添加新的CSI数据到缓冲区
    for (int i = 0; i < length && CSI_Q_INDEX < CSI_BUFFER_LENGTH; i++) {
        CSI_Q[CSI_Q_INDEX++] = (int16_t)csi_data[i];
    }
    
    // 只在要处理算法时输出缓冲区状态日志
    if (should_process) {
        ESP_LOGI(TAG, "CSI Buffer Status: %d samples stored", CSI_Q_INDEX);
    }
    
    // 如果不需要处理算法，就直接返回（不输出日志）
    if (!should_process) {
        return;
    }
    
    // 更新上次处理时间
    last_process_time = current_time;

    if (mqtt_connected == false) {
        ESP_LOGW(TAG, "MQTT client not connected");
        return;
    }

    // 以下是算法处理部分，保留日志输出
    // 1. 填写团队信息

    // 3036382909 Bai Junhao
    // 3036380559 Long Qian
    // 3036380004 Shi Xianjie
    // 3036414817 Wei Shuang

    ESP_LOGI(TAG, "================ GROUP INFO ================");
    const char *TEAM_MEMBER[] = {"Bai Junhao", "Long Qian", "Shi Xianjie", "Wei Shuang"};
    const char *TEAM_UID[] = {"3036382909", "3036380559", "3036380004", "3036414817"};
    ESP_LOGI(TAG, "TEAM_MEMBER: %s, %s, %s, %s | TEAM_UID: %s, %s, %s, %s",
                TEAM_MEMBER[0], TEAM_MEMBER[1], TEAM_MEMBER[2], TEAM_MEMBER[3],
                TEAM_UID[0], TEAM_UID[1], TEAM_UID[2], TEAM_UID[3]);
    ESP_LOGI(TAG, "================ END OF GROUP INFO ================");

    // 2. 调用算法函数
    int flag_motion = 0;
    int flag_breath = 0;

    bool motion = motion_detection(&flag_motion);
    ESP_LOGI(TAG, "Motion Detection Result: %s", motion ? "MOTION DETECTED" : "NO MOTION");

    int bpm = 0;
    // 只有当有足够的数据时才进行呼吸率估计
    if (CSI_Q_INDEX >= CSI_BUFFER_LENGTH) {
        bpm = breathing_rate_estimation(&flag_breath);
        ESP_LOGI(TAG, "Estimated Breathing Rate: %d BPM", bpm);
    } else {
        ESP_LOGW(TAG, "Not enough CSI data for breathing estimation: %d/%d", 
                 CSI_Q_INDEX, CSI_BUFFER_LENGTH);
    }

    // 发送MQTT消息
    mqtt_send(motion, bpm);
}


//------------------------------------------------------CSI Config Initialize------------------------------------------------------
static void wifi_csi_init()
{   
    esp_task_wdt_add(NULL); // Add the current task to the watchdog;
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    wifi_csi_config_t csi_config = {
        .enable                   = true,                           
        .acquire_csi_legacy       = false,               
        .acquire_csi_force_lltf   = false,           
        .acquire_csi_ht20         = true,                  
        .acquire_csi_ht40         = true,                  
        .acquire_csi_vht          = false,                  
        .acquire_csi_su           = false,                   
        .acquire_csi_mu           = false,                   
        .acquire_csi_dcm          = false,                  
        .acquire_csi_beamformed   = false,           
        .acquire_csi_he_stbc_mode = 2,                                                                                                                                                                                                                                                                               
        .val_scale_cfg            = 0,                    
        .dump_ack_en              = false,                      
        .reserved                 = false                         
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

//------------------------------------------------------Main Function------------------------------------------------------
void app_main()
{
    /**
     * @brief Initialize NVS
     */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /**
     * @brief Initialize Wi-Fi
     */
    wifi_init();

    // Get Device MAC Address
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Device MAC Address: " MACSTR, MAC2STR(mac));

    // Try to connect to WiFi
    ESP_LOGI(TAG, "Connecting to WiFi...");

    // Create a semaphore to signal when we're connected and have an IP
    static bool ip_obtained = false;
    int retry_count = 0;
    const int max_retries = 30; // Increase retry count to 30 seconds
    // Wait for Wi-Fi connection and IP assignment
    while (!ip_obtained && retry_count < max_retries) {
        // Check if we're connected to WiFi
        wifi_ap_record_t ap_info;
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            // Check if we have an IP address
            esp_netif_ip_info_t ip_info;
            esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
            if (netif != NULL && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
                // Check if IP is valid (not 0.0.0.0)
                if (ip_info.ip.addr != 0) {
                    ESP_LOGI(TAG, "Successfully connected to AP: %s", ap_info.ssid);
                    ESP_LOGI(TAG, "IP Address: " IPSTR, IP2STR(&ip_info.ip));
                    ip_obtained = true;
                    break;
                }
            }
        }
        
        ESP_LOGI(TAG, "Waiting for Wi-Fi connection and IP assignment... (%d/%d)", retry_count + 1, max_retries);
        retry_count++;
        vTaskDelay(pdMS_TO_TICKS(1000)); // Wait for 1 second
    }

    if (!ip_obtained) {
        ESP_LOGE(TAG, "Failed to connect to WiFi and obtain IP address after %d seconds", max_retries);
        ESP_LOGI(TAG, "Restarting system in 5 seconds...");
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
        return;
        
    }

    /**
     * @brief Initialize ESP-NOW
     */

    if (wifi_connected) {
        esp_now_peer_info_t peer = {
            .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
            .ifidx     = WIFI_IF_STA,
            .encrypt   = false,
            .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
        };

        wifi_esp_now_init(peer); // Initialize ESP-NOW Communication
        wifi_csi_init(); // Initialize CSI Collection

        // Initialize MQTT after WiFi is connected
        mqtt_init();
    } else {
        ESP_LOGI(TAG, "WiFi connection failed");
        return;
    }
}
