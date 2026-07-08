import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedDogEmotionCRNN(nn.Module):
    def __init__(self, num_classes=5, input_channels=1, n_mels=64):
        super(ImprovedDogEmotionCRNN, self).__init__()
        
        # -------------------------------------------
        # 1. CNN 空间特征提取 (4个卷积块 + 非对称池化)
        # -------------------------------------------
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 32
            self._make_block(input_channels, 32),
            # Block 2: 32 -> 64
            self._make_block(32, 64),
            # Block 3: 64 -> 128
            self._make_block(64, 128),
            # Block 4: 128 -> 256
            self._make_block(128, 256)
        )
        
        # -------------------------------------------
        # 2. BiLSTM 时序动态建模
        # -------------------------------------------
        # 初始 n_mels 默认为 64。经过 4 次高度减半的池化: 64 -> 32 -> 16 -> 8 -> 4
        # 最终 CNN 输出的通道数为 256，高度为 4。
        rnn_input_size = 256 * (n_mels // 16) 
        
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=256, 
                           num_layers=2, batch_first=True, bidirectional=True)
                           
        # -------------------------------------------
        # 3. Attention 注意力机制 (选择性听觉)
        # -------------------------------------------
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # -------------------------------------------
        # 4. 语义转译输出前置
        # -------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 严格对齐 PPT
            nn.Linear(512, num_classes)
        )

    def _make_block(self, in_ch, out_ch):
        """辅助函数：构建带有非对称池化的卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # 核心创新：非对称池化，只压缩频率维 (2)，保留时间维 (1)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) 
        )

    def forward(self, x, return_attention=False):
        # x shape: [batch, 1, n_mels, time_steps]
        
        # 1. CNN
        x = self.cnn(x) 
        
        # 2. 变形对接 BiLSTM
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # [batch, time, channels, freq]
        x = x.view(batch_size, time, channels * freq) # [batch, time, 1024]
        
        # 3. BiLSTM
        out, _ = self.rnn(x) # out shape: [batch, time, 512]
        
        # 4. Attention
        att_weights = self.attention(out) # [batch, time, 1]
        att_weights = F.softmax(att_weights, dim=1) # [batch, time, 1]
        
        # 加权求和生成全局上下文向量
        context = torch.bmm(out.transpose(1, 2), att_weights).squeeze(2) # [batch, 512]
        
        # 5. 分类输出
        logits = self.classifier(context) # [batch, num_classes]
        
        # 如果是供 Gradio 网页端使用，则同时返回注意力权重画热力图
        if return_attention:
            return logits, att_weights
            
        return logits