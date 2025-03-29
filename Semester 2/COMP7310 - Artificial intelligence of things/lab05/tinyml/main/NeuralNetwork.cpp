#include "NeuralNetwork.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include <stdlib.h>

const int kArenaSize = 20000;

NeuralNetwork::NeuralNetwork() {
    error_reporter = new tflite::MicroErrorReporter();

    // Load the TFLite model from model_data.h
    model = tflite::GetModel(imu_model_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();
    resolver->AddSoftmax();

    tensor_arena = (uint8_t *)malloc(kArenaSize);
    if (!tensor_arena) {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    interpreter = new tflite::MicroInterpreter(model, *resolver, tensor_arena, kArenaSize);
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    input = interpreter->input(0);
    output = interpreter->output(0);
}

float *NeuralNetwork::getInputBuffer() {
    return input->data.f;
}

float *NeuralNetwork::getOutputBuffer() {
    return output->data.f;
}

float NeuralNetwork::predict() {
    interpreter->Invoke();
    return output->data.f[0];
}
