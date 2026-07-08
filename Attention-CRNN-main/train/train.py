import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


from tensorboardX import SummaryWriter

# 其他文件中导入数据集和模型
from dataset import DogEmotionDataset
from model import ImprovedDogEmotionCRNN

def train():
    # ================= 1. 环境与配置 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备进行训练: {device}")
    
    batch_size = 32
    epochs = 30
    lr = 0.001
    
    # 你的数据路径 (如果需要，请微调这里的绝对/相对路径)
    train_dir = r'data_mel/train'
    val_dir = r'data_mel/val'
    
    # 日志记录器
    writer = SummaryWriter('runs/Improved_CRNN_Exp')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # ================= 2. 数据加载与类别权重 =================
    train_dataset = DogEmotionDataset(train_dir)
    val_dataset = DogEmotionDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 根据你之前截图的数据量计算权重 (angry, anxious, happy, lonely, sad)
    counts = [1200, 1012, 775, 604, 639]
    weights = torch.FloatTensor([sum(counts)/(len(counts)*c) for c in counts]).to(device)
    print("⚖️ 已启用类别权重:", weights.tolist())
    
    # ================= 3. 模型与优化器 =================
    model = ImprovedDogEmotionCRNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_acc = 0.0
    
    # ================= 4. 训练主循环 =================
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # --- 训练阶段 ---
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * inputs.size(0)
            t_correct += (outputs.argmax(1) == labels).sum().item()
            t_total += labels.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # --- 验证阶段 ---
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss += criterion(outputs, labels).item() * inputs.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)
        
        # --- 计算与记录指标 ---
        train_l, train_a = t_loss/t_total, 100.*t_correct/t_total
        val_l, val_a = v_loss/v_total, 100.*v_correct/v_total
        
        print(f"Train Loss: {train_l:.4f} | Train Acc: {train_a:.2f}%")
        print(f"Val Loss:   {val_l:.4f} | Val Acc:   {val_a:.2f}%")
        
        writer.add_scalars('Loss', {'train': train_l, 'val': val_l}, epoch)
        writer.add_scalars('Accuracy', {'train': train_a, 'val': val_a}, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        history['train_loss'].append(train_l)
        history['val_loss'].append(val_l)
        history['train_acc'].append(train_a)
        history['val_acc'].append(val_a)
        
        scheduler.step(val_l)
        
        if val_a > best_acc:
            best_acc = val_a
            torch.save(model.state_dict(), 'best_improved_model.pth')
            print(f"🌟 发现新最佳模型，已保存！(准确率: {best_acc:.2f}%)")

    # ================= 5. 训练结束与绘图 =================
    writer.close()
    print("\n🎉 训练完成！正在生成曲线图...")
    plot_curves(history, epochs)

def plot_curves(history, epochs):
    """绘制训练曲线并保存"""
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train')
    plt.plot(epochs_range, history['val_loss'], label='Val')
    plt.title('Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train')
    plt.plot(epochs_range, history['val_acc'], label='Val')
    plt.title('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_plot.png', dpi=300)
    print("📊 曲线图已保存至 'improved_training_plot.png'")

if __name__ == "__main__":
    train()