#ifndef __NeuralNetwork__
#define __NeuralNetwork__

#include <stdint.h>

namespace tflite {
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
}

struct TfLiteTensor;

class NeuralNetwork {
private:
    tflite::MicroMutableOpResolver<10> *resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *tensor_arena;

public:
    // Returns a pointer to the input buffer (float array)
    float *getInputBuffer();
    // Returns a pointer to the output buffer (float array)
    float *getOutputBuffer();

    NeuralNetwork();
    // Runs inference (output is read via getOutputBuffer)
    float predict();
};

#endif
