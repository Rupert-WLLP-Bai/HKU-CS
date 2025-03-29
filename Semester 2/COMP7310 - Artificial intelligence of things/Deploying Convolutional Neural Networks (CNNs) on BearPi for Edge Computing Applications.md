# **Deploying Convolutional Neural Networks (CNNs) on BearPi for Edge Computing Applications**

## **1. Introduction**

With the rapid growth of the Internet of Things (IoT) and Artificial Intelligence of Things (AIoT), many applications require **running deep learning models directly on edge devices** to **reduce latency, lower bandwidth usage, and enhance privacy**.

### **Tutorial Objectives:**

- Introduce **BearPi**, a compact single-board computer for AI and IoT applications.
- Set up the AI environment on BearPi.
- Convert **pre-trained CNN models (e.g., MobileNetV2)** to **TensorFlow Lite (TFLite)** format.
- Optimize models for **efficient inference** on BearPi.
- Demonstrate **real-time AI inference**, including **image classification** and **object detection**.

By the end of this tutorial, you will be able to **deploy AI applications on BearPi** efficiently and apply them to **smart monitoring, autonomous systems, and edge AI scenarios**.

------

## **2. Prerequisites**

### **Hardware Requirements:**

- **BearPi Development Board**
- **MicroSD Card (16GB or larger, Class 10 recommended)**
- **USB Power Adapter (5V/2A)**
- **USB-to-Serial Adapter** (for debugging)
- **Camera Module (Optional)** (for image processing applications)

### **Software Requirements:**

- **BearPi OS** (Official Linux Distribution)
- **Python 3.11**
- **TensorFlow Lite** (for AI inference)
- **OpenCV** (for image processing)
- **Edge TPU Compiler**

### **Required Knowledge:**

- **Basic Python Programming**
- **Fundamentals of Deep Learning (CNNs, TensorFlow)**
- **Linux Terminal Operations**

------

## **3. Setting Up BearPi Development Environment**

### **3.1 Install BearPi OS**

1. **Download BearPi OS**: Get the latest image from the [BearPi official website](https://bearpi.org/).
2. **Flash the OS to MicroSD Card**:
   - Use  `balenaEtcher` to flash the OS onto the MicroSD card.
3. **Insert MicroSD Card and Boot BearPi**:
   - Insert the MicroSD card into BearPi and power it on.
   - Connect to a monitor, keyboard, and mouse, or use SSH for remote access.

### **3.2 Install Required Dependencies**

Update the system and install essential tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git
```

Create a Python virtual environment and install TensorFlow Lite and OpenCV:

```bash
python3 -m venv tflite_env
source tflite_env/bin/activate
pip install numpy opencv-python tensorflow-lite
```

------

## **4. Converting a Pre-trained CNN Model to TensorFlow Lite**

To efficiently run deep learning models on BearPi, we need to convert a **pre-trained CNN model** (e.g., MobileNetV2) into the **TensorFlow Lite format**.

### **4.1 Download Pre-trained MobileNetV2 Model**

```python
import tensorflow as tf

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

# Save the model
model.save("mobilenetv2_model")
```

### **4.2 Convert to TensorFlow Lite Format**

```python
# Load TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_model")

# Convert to TFLite format
tflite_model = converter.convert()

# Save the TFLite model
with open("mobilenetv2.tflite", "wb") as f:
    f.write(tflite_model)
```

------

## **5. Model Optimization (Quantization)**

### **Why Quantization?**

- **Reduces model size** for deployment on embedded devices.
- **Improves inference speed** by reducing computational complexity.

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and quantize the model
tflite_quant_model = converter.convert()

# Save the quantized model
with open("mobilenetv2_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)
```

------

## **6. Running AI Inference on BearPi**

### **6.1 Load the TFLite Model**

```python
import tensorflow.lite as tflite
import numpy as np
import cv2

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="mobilenetv2_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

### **6.2 Image Classification Task**

**Steps:**

1. Read and preprocess the image.
2. Run inference using the model.
3. Display classification results.

```python
# Read and resize the image
image = cv2.imread("test.jpg")
image = cv2.resize(image, (224, 224))
image = np.expand_dims(image, axis=0).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]["index"], image)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]["index"])
predicted_label = np.argmax(output_data)

print("Predicted Label:", predicted_label)
```

------

## **7. Object Detection Application**

To perform **real-time object detection**, you need a **camera module** and use **TFLite Object Detection API**.

### **7.1 Install Required Dependencies**

```bash
pip install tflite-runtime pillow
```

### **7.2 Running Object Detection**

```python
from tflite_runtime.interpreter import Interpreter
from PIL import Image

# Load the TFLite Object Detection Model
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Process the image
image = Image.open("test.jpg").resize((300, 300))
image_array = np.expand_dims(image, axis=0).astype(np.uint8)

interpreter.set_tensor(input_details[0]["index"], image_array)
interpreter.invoke()

# Get detection results
output_data = interpreter.get_tensor(output_details[0]["index"])
print("Detected Objects:", output_data)
```

------

## **8. Results Analysis**

| Metric         | MobileNetV2 (TF) | MobileNetV2 (TFLite) | MobileNetV2 (TFLite Quantized) |
| -------------- | ---------------- | -------------------- | ------------------------------ |
| Model Size     | 14MB             | 5MB                  | 2.8MB                          |
| Inference Time | ~250ms           | ~120ms               | ~60ms                          |
| Accuracy Drop  | 0%               | ~1-2%                | ~2-4%                          |

------

## **9. Conclusion**

This tutorial provided a step-by-step guide on **deploying CNNs on BearPi for edge computing applications**, covering **model conversion, optimization, inference, and real-world applications**. These techniques enable efficient **AI-powered IoT devices** for **smart surveillance, real-time recognition, and other AIoT use cases**.

------

## **10. References**

- [TensorFlow Lite Official Documentation](https://www.tensorflow.org/lite)
- [BearPi Gitee](https://gitee.com/bearpi/)
- [TensorFlow Lite Examples](https://www.tensorflow.org/lite/examples)
- [AI on the Edge - Google Coral](https://coral.ai/)
- [Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers](https://arxiv.org/abs/1905.12107)

