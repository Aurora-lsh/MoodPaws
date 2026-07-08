# MoodPaws

**基于多模态情绪转义的智能宠物寄养屋**

[English Version](./README_EN.md)

## 项目简介

MoodPaws 是一个基于多模态情绪转义的智能宠物寄养屋，旨在提升宠物寄养体验。系统通过物联网技术、AI 行为分析和情感识别，结合智能项圈和寄养屋中的传感器，实时监控宠物的生理状态、行为和情绪变化。通过情绪识别，MoodPaws 将宠物的情绪转化为主人能理解的反馈，实现情感化互动，同时主人可通过 APP 远程观察宠物并进行实时互动，缓解分离焦虑。

## 核心特性

- **全场景主动感知** — 寄养屋 + 智能项圈双重传感器网络（温湿度、重量、GPS、心率、血氧等）
- **环境自适应** — 自动维持最适宜的温湿度，恒温加热、通风散热联动控制
- **AI 深度行为理解** — YOLO-pose 姿态检测 + ImprovedDogEmotionCRNN 情绪识别
- **全周期健康预警** — LSTM + DBSCAN 时序分析，异常检测与健康预测
- **远程沉浸式交互** — APP 实时监控、音视频互动、"语音翻译"对话
- **一站式智慧管理** — Web 管理平台支持集中监控、预警中心、宠物数字档案
- **AIGC 内容生成** — DeepSeek 大模型生成拟人化宠物日常故事与情绪报告

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      云端 (阿里云 IoT)                    │
│  设备管理 · 数据存储 · AI预测(LSTM/DBSCAN) · DeepSeek LLM │
└──────────────────────┬──────────────────────────────────┘
                       │ MQTT
┌──────────────────────┴──────────────────────────────────┐
│                   边缘侧 (RDK X5)                        │
│  YOLO11n-pose · PetEmotionCRNN · 本地自治决策             │
└──────────┬──────────────────────────────┬───────────────┘
           │ UART/Wi-Fi                   │ UART/Wi-Fi
┌──────────┴──────────┐    ┌──────────────┴───────────────┐
│    智能寄养屋         │    │       智能项圈                │
│  STM32 + 传感器阵列   │    │  STM32 + MAX30102 + GPS/IMU │
│  DHT22/MQ135/HX711   │    │  DS18B20 + ESP8266          │
│  舵机/风扇/PTC加热    │    │                              │
└─────────────────────┘    └──────────────────────────────┘
```

## 项目结构

```
项目源码/
├── APP-web-server/
│   ├── app/                  # 移动端 APP（Vue 3 + Capacitor）
│   ├── moodpaws-server/      # 后端服务（Express + MQTT + SQLite）
│   └── web/                  # Web 管理端（React + TypeScript）
├── Attention-CRNN-main/      # 宠物情绪识别模型（CNN + BiLSTM + Attention）
│   ├── train/                # 模型训练代码
│   ├── app.py                # 模型推理服务
│   └── best_model.pth        # 训练好的模型权重
└── YOLO-Animal-Pose-main/    # 宠物姿态检测模型（YOLO11n-pose）
    ├── model/                # 模型文件
    ├── YOLO-Animal-Pose-main.py  # 姿态检测推理
    └── pt_onnx.py            # PyTorch → ONNX 模型转换
```

## 技术栈

| 层级 | 技术 |
|------|------|
| **前端 APP** | Vue 3、Capacitor、ECharts、Day.js |
| **Web 管理端** | React 18、TypeScript、Vite |
| **后端服务** | Node.js、Express、MQTT、SQLite |
| **AI 视觉** | YOLO11n-pose、ONNX、INT8 量化、NPU 加速 |
| **AI 语音** | ImprovedDogEmotionCRNN (CNN + BiLSTM + Attention)、librosa |
| **AI 分析** | LSTM、DBSCAN、Isolation Forest、LSTM-AutoEncoder |
| **大模型** | DeepSeek LLM、讯飞 TTS |
| **物联网** | STM32、ESP8266、阿里云 IoT 平台、MQTT |
| **传感器** | MAX30102、DHT22、DS18B20、HX711、MQ-135、GPS + IMU |

## 快速开始

### 前置要求

- Node.js >= 18
- pnpm 或 npm
- Python >= 3.8（用于 AI 模型）
- 阿里云 IoT 平台账号

### 启动后端服务

```bash
cd APP-web-server/moodpaws-server
cp .env.example .env
# 编辑 .env 填入 MQTT 等配置
npm install
npm run dev
```

### 启动移动端 APP

```bash
cd APP-web-server/app
cp .env.example .env.local
# 编辑 .env.local 填入服务端地址
npm install
npm run dev
```

### 启动 Web 管理端

```bash
cd APP-web-server/web
npm install
npm run dev
```

### 运行 AI 模型

```bash
# 宠物情绪识别
cd Attention-CRNN-main
pip install torch librosa numpy
python app.py

# 宠物姿态检测
cd YOLO-Animal-Pose-main
pip install ultralytics opencv-python
python YOLO-Animal-Pose-main.py
```

## 硬件传感器模块

| 模块 | 型号 | 功能 |
|------|------|------|
| 心率血氧 | MAX30102 | 宠物心率与血氧饱和度连续监测 |
| 运动定位 | GPS + 九轴 IMU | 室内外无缝定位与运动轨迹追踪 |
| 体表温度 | DS18B20 | 宠物体表温度检测 |
| 环境温湿度 | DHT22 | 寄养屋环境温湿度监测 |
| 空气质量 | 三合一气体模块 + MQ-135 | 氨气/硫化氢/CO2 检测与异味监测 |
| 体重检测 | HX711 | 宠物体重动态估计 |
| 通风散热 | PWM 风扇 | 温度/空气质量联动通风 |
| 恒温加热 | PTC 加热片 | 自限温安全加热 |
| 逗宠交互 | 舵机 + 逗猫棒 | 自动/远程逗宠互动 |

## AI 算法性能

| 指标 | 结果 |
|------|------|
| 宠物品种识别准确率 | 94.5% |
| 宠物姿态识别准确率 | 91.2% |
| 情绪识别准确率（视觉+声音） | 92.3% |
| 视觉推理帧率 (RDK X5 NPU) | 35-42 FPS |
| 语义转译端到端延迟 | < 2.5s |

## 许可证

本项目为参加中国大学生计算机设计大赛的参赛作品。

## 参考文献

- 洞见研究院. 2025年中国宠物消费趋势白皮书
- Gong Y, et al. AST: Audio Spectrogram Transformer. arXiv:2104.01778, 2021
- Pawar M D, et al. CNN based automatic speech emotion recognition using MFCC. Multimedia Tools and Applications, 2021
- Chavez-Guerrero V O, et al. Classification of Domestic Dogs Emotional Behavior Using Computer Vision. Computacion y Sistemas, 2022
