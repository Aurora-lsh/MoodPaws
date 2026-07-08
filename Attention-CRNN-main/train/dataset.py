#该文件定义了一个 PyTorch 数据集类 `DogEmotionDataset`，用于加载和处理狗的情绪数据。数据集从指定目录中读取预处理后的梅尔频谱特征（以 .npy 格式保存）和对应的情绪标签，并将它们转换为适合模型训练的格式。每个样本返回一个形状为 [1, 64, 150] 的梅尔频谱张量和一个整数标签。
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class DogEmotionDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: 处理后的数据路径，例如 'data_mel/train'
        """
        self.data_dir = data_dir
        self.filepaths = []
        self.labels = []
        
        # 定义类别映射 (严格对应你的 5 个情绪类别)
        self.label_map = {'angry': 0, 'anxious': 1, 'happy': 2, 'lonely': 3, 'sad': 4}
        
        # 遍历目录收集所有 .npy 文件路径和对应的标签
        for emotion in os.listdir(data_dir):
            if emotion not in self.label_map:
                continue
            emotion_dir = os.path.join(data_dir, emotion)
            
            for filename in os.listdir(emotion_dir):
                if filename.endswith('.npy'):
                    self.filepaths.append(os.path.join(emotion_dir, filename))
                    self.labels.append(self.label_map[emotion])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # 1. 加载通过 numpy 保存的梅尔频谱特征
        mel_spec = np.load(self.filepaths[idx])
        
        # 2. 转换为 PyTorch Tensor，并增加一个 Channel 维度 (变成 [1, 64, 150])
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        
        # 3. 转换标签为 LongTensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return mel_spec, label