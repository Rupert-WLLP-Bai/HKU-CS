idf_component_register(SRCS main.cpp model_data.cc NeuralNetwork.cpp
                       PRIV_REQUIRES spi_flash esp-tflite-micro esp_timer
                       INCLUDE_DIRS "."
                       )
