# MoodPaws

**An Intelligent Pet Boarding House Based on Multimodal Emotion Translation**

[中文版本](./README.md)

## Introduction

MoodPaws is an intelligent pet boarding house powered by multimodal emotion translation, designed to enhance the pet boarding experience. The system leverages IoT technology, AI-based behavior analysis, and emotion recognition, combined with a smart collar and in-house sensors, to monitor pets' physiological status, behavior, and emotional changes in real time. Through emotion recognition, MoodPaws translates pet emotions into human-understandable feedback for emotional interaction. Pet owners can remotely observe and interact with their pets via the mobile APP, alleviating separation anxiety.

## Key Features

- **Full-Scenario Active Sensing** — Dual sensor network (boarding house + smart collar) covering temperature, humidity, weight, GPS, heart rate, SpO2, and more
- **Adaptive Environment Control** — Automatic temperature and humidity regulation with thermostat heating and ventilation联动 control
- **AI-Powered Behavior Understanding** — YOLO-pose for posture detection + ImprovedDogEmotionCRNN for emotion recognition
- **Full-Cycle Health Monitoring** — LSTM + DBSCAN time-series analysis for anomaly detection and health prediction
- **Remote Immersive Interaction** — Real-time monitoring via APP, audio/video interaction, and "voice translation" dialogue
- **One-Stop Smart Management** — Web dashboard with centralized monitoring, alert center, and digital pet profiles
- **AIGC Content Generation** — DeepSeek LLM generates anthropomorphic daily stories and emotion reports for pets

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Cloud (Alibaba Cloud IoT)              │
│  Device Mgmt · Data Storage · AI Prediction · DeepSeek   │
└──────────────────────┬──────────────────────────────────┘
                       │ MQTT
┌──────────────────────┴──────────────────────────────────┐
│                    Edge (RDK X5)                         │
│  YOLO11n-pose · PetEmotionCRNN · Local Decision Making   │
└──────────┬──────────────────────────────┬───────────────┘
           │ UART/Wi-Fi                   │ UART/Wi-Fi
┌──────────┴──────────┐    ┌──────────────┴───────────────┐
│  Smart Boarding House│    │        Smart Collar           │
│  STM32 + Sensor Array│    │  STM32 + MAX30102 + GPS/IMU  │
│  DHT22/MQ135/HX711  │    │  DS18B20 + ESP8266           │
│  Servo/Fan/PTC Heat  │    │                              │
└─────────────────────┘    └──────────────────────────────┘
```

## Project Structure

```
Project Root/
├── APP-web-server/
│   ├── app/                  # Mobile APP (Vue 3 + Capacitor)
│   ├── moodpaws-server/      # Backend Server (Express + MQTT + SQLite)
│   └── web/                  # Web Management Dashboard (React + TypeScript)
├── Attention-CRNN-main/      # Pet Emotion Recognition Model (CNN + BiLSTM + Attention)
│   ├── train/                # Model Training Code
│   ├── app.py                # Model Inference Service
│   └── best_model.pth        # Trained Model Weights
└── YOLO-Animal-Pose-main/    # Pet Pose Detection Model (YOLO11n-pose)
    ├── model/                # Model Files
    ├── YOLO-Animal-Pose-main.py  # Pose Detection Inference
    └── pt_onnx.py            # PyTorch → ONNX Model Conversion
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Mobile APP** | Vue 3, Capacitor, ECharts, Day.js |
| **Web Dashboard** | React 18, TypeScript, Vite |
| **Backend Server** | Node.js, Express, MQTT, SQLite |
| **AI Vision** | YOLO11n-pose, ONNX, INT8 Quantization, NPU Acceleration |
| **AI Audio** | ImprovedDogEmotionCRNN (CNN + BiLSTM + Attention), librosa |
| **AI Analytics** | LSTM, DBSCAN, Isolation Forest, LSTM-AutoEncoder |
| **LLM** | DeepSeek LLM, iFlytek TTS |
| **IoT** | STM32, ESP8266, Alibaba Cloud IoT Platform, MQTT |
| **Sensors** | MAX30102, DHT22, DS18B20, HX711, MQ-135, GPS + IMU |

## Getting Started

### Prerequisites

- Node.js >= 18
- pnpm or npm
- Python >= 3.8 (for AI models)
- Alibaba Cloud IoT Platform account

### Start Backend Server

```bash
cd APP-web-server/moodpaws-server
cp .env.example .env
# Edit .env with your MQTT credentials and other config
npm install
npm run dev
```

### Start Mobile APP

```bash
cd APP-web-server/app
cp .env.example .env.local
# Edit .env.local with your server address
npm install
npm run dev
```

### Start Web Dashboard

```bash
cd APP-web-server/web
npm install
npm run dev
```

### Run AI Models

```bash
# Pet Emotion Recognition
cd Attention-CRNN-main
pip install torch librosa numpy
python app.py

# Pet Pose Detection
cd YOLO-Animal-Pose-main
pip install ultralytics opencv-python
python YOLO-Animal-Pose-main.py
```

## Hardware Sensor Modules

| Module | Model | Function |
|--------|-------|----------|
| Heart Rate & SpO2 | MAX30102 | Continuous heart rate and blood oxygen monitoring |
| Motion Tracking | GPS + 9-axis IMU | Seamless indoor/outdoor positioning and trajectory tracking |
| Body Temperature | DS18B20 | Pet body surface temperature detection |
| Environmental T&H | DHT22 | Ambient temperature and humidity monitoring |
| Air Quality | 3-in-1 Gas Module + MQ-135 | Ammonia/H₂S/CO₂ detection and odor monitoring |
| Weight Estimation | HX711 | Dynamic pet weight estimation |
| Ventilation | PWM Fan | Temperature/air quality linked ventilation |
| Thermostat Heating | PTC Heater | Self-regulating safe heating |
| Pet Interaction | Servo + Cat Teaser | Automatic and remote pet play interaction |

## AI Algorithm Performance

| Metric | Result |
|--------|--------|
| Pet Breed Recognition Accuracy | 94.5% |
| Pet Pose Recognition Accuracy | 91.2% |
| Emotion Recognition Accuracy (Vision + Audio) | 92.3% |
| Vision Inference FPS (RDK X5 NPU) | 35-42 FPS |
| Semantic Translation End-to-End Latency | < 2.5s |

## License

This project is a competition entry for the China Collegiate Computing Contest (中国大学生计算机设计大赛).

## References

- Insight Research Institute. 2025 China Pet Consumption Trends White Paper
- Gong Y, et al. AST: Audio Spectrogram Transformer. arXiv:2104.01778, 2021
- Pawar M D, et al. CNN based automatic speech emotion recognition using MFCC. Multimedia Tools and Applications, 2021
- Chavez-Guerrero V O, et al. Classification of Domestic Dogs Emotional Behavior Using Computer Vision. Computacion y Sistemas, 2022
