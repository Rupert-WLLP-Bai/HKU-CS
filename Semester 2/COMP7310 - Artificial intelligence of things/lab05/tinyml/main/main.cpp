#include "NeuralNetwork.h"
#include "selected_data.h"
#include <cstdint>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "esp_timer.h" // add esp_timer to CMakeLists.txt PRIV_REQUIRES

static const char *TAG = "MAIN";
static const char *gesture_names[6] = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

extern "C" void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_INFO);
    ESP_LOGI(TAG, "Starting Neural Network...");

    // Create an instance of the neural network
    NeuralNetwork *nn = new NeuralNetwork();

    int numSamples = NUM_SAMPLES;
    int correctCount = 0;

    // TODO: Add a timer to record the inference time (13 pts)
    // YOUR CODE HERE

    int64_t totalInferenceTime = 0; // Variable to store total inference time

    for (int i = 0; i < numSamples; i++) {
        float *inputBuffer = nn->getInputBuffer();
        for (int j = 0; j < FEATURE_SIZE; j++) {
            inputBuffer[j] = X_selected[i][j];
        }
        
        int64_t startTime = esp_timer_get_time(); // Start time for inference

        nn->predict();

        int64_t endTime = esp_timer_get_time(); // End time for inference
        int64_t inferenceTime = endTime - startTime; // Calculate inference time
        totalInferenceTime += inferenceTime; // Accumulate inference time

        float *outputBuffer = nn->getOutputBuffer();
        int predictedLabel = 0;
        float maxProb = outputBuffer[0];
        for (int k = 1; k < 6; k++) {
            if (outputBuffer[k] > maxProb) {
                maxProb = outputBuffer[k];
                predictedLabel = k;
            }
        }

        int trueLabel = y_selected[i];
        if (predictedLabel == trueLabel) {
            correctCount++;
        }

        ESP_LOGI(TAG, "Sample %d: GT=%d, GT_gesture=%s, Predicted=%d, Predicted_gesture=%s",
                 i, trueLabel, gesture_names[trueLabel],
                 predictedLabel, gesture_names[predictedLabel]);
    }

    // Calculate average inference time
    float averageInferenceTime = static_cast<float>(totalInferenceTime) / numSamples;
    ESP_LOGI(TAG, "Average Inference Time: %.5f seconds", averageInferenceTime / 1e6);
    ESP_LOGI(TAG, "Total Inference Time: %.5f seconds", totalInferenceTime / 1e6);

    // END OF YOUR CODE

    float accuracy = (float)correctCount / numSamples;
    ESP_LOGI(TAG, "Final Accuracy: %.5f%%", accuracy * 100);

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}