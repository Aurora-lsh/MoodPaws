import os
import time
import shutil
import io
import base64
import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区域与全局初始化
# ==========================================
MODEL_PATH = r"best_model.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEEPSEEK_API_KEY = "sk-" # 替换为你的真实 API Key
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

EMOTION_MAP = {
    0: "愤怒 (Angry)", 1: "焦虑 (Anxious)", 2: "开心 (Happy)", 
    3: "孤独 (Lonely)", 4: "悲伤 (Sad)"
}

# 阿里云 IoT 对接编码映射
EMOTION_CODE_MAP = {
    "愤怒 (Angry)": 1, "焦虑 (Anxious)": 2, "开心 (Happy)": 3, 
    "孤独 (Lonely)": 4, "悲伤 (Sad)": 5
}

# 初始化本地文件队列目录
QUEUE_DIR = "data_queue"
os.makedirs(QUEUE_DIR, exist_ok=True)
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# 加载 PyTorch 模型
from model import ImprovedDogEmotionCRNN
print(f"正在加载模型至设备: {DEVICE}...")
try:
    model = ImprovedDogEmotionCRNN(num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")

# ==========================================
# 2. 核心逻辑函数
# ==========================================
def get_llm_translation(emotion_label):
    try:
        prompt = f"你现在是一只小狗。你当前的情绪是：【{emotion_label}】。请用一句简短的话（15字以内）表达你的想法。语气生动。只输出台词内容。"
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个智能狗狗翻译器。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=50
        )
        return response.choices[0].message.content.strip().replace('"', '').replace('“', '')
    except:
        return "汪汪！(翻译官网络不好...)"

def generate_spectrogram_base64(mel_spec_db):
    plt.figure(figsize=(4, 1)) 
    plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def process_audio_logic(audio_path):
    # 1. 预处理
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    TARGET_FRAMES = 150
    if mel_spec_db.shape[1] < TARGET_FRAMES:
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, TARGET_FRAMES - mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :TARGET_FRAMES]

    # 2. 推理
    current_device = next(model.parameters()).device
    input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(current_device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    # 3. 解析结果
    emotion_label = EMOTION_MAP.get(pred_idx.item(), "未知")
    confidence = f"{conf.item() * 100:.2f}%"
    
    # ==========================================
    # 写入本地文件队列，供云端程序读取
    # ==========================================
    label_code = EMOTION_CODE_MAP.get(emotion_label, 0)
    # 使用毫秒级时间戳防止文件名冲突
    current_timestamp = int(time.time() * 1000) 
    try:
        filename = f"task_{current_timestamp}.txt"
        filepath = os.path.join(QUEUE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{current_timestamp},{label_code}")
    except Exception as e:
        print(f"写入本地队列失败: {e}")
    # ==========================================

    translation = get_llm_translation(emotion_label)
    spectrogram_b64 = generate_spectrogram_base64(mel_spec_db)

    return emotion_label, confidence, translation, spectrogram_b64

# ==========================================
# 3. FastAPI 路由
# ==========================================
app = FastAPI(title="MoodPaws API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/api/analyze")
async def analyze_audio(audio_file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, f"{int(time.time())}_{audio_file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        emotion, conf, speech, spec_b64 = await run_in_threadpool(process_audio_logic, temp_path)
        
        return {
            "status": "success", "emotion": emotion, "confidence": conf,
            "translation": speech, "spectrogram": spec_b64
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    print("MoodPaws 主后端已启动！监听 http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)