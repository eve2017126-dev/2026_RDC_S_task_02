import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchvision import transforms

from dataset import BusCarDataset
from model import CNNImproved
from utils import plot_training_history

def train_focused_on_bus():
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"PyTorch CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs("../models", exist_ok=True)

    # 定义数据增强 - 对巴士类别使用更强的增强
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),  # 增加旋转幅度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 加载数据集
    dataset = BusCarDataset("../data/train", transform=train_transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 初始化模型
    model = CNNImproved(num_classes=2, input_size=64).to(device)
    print("使用改进版CNN模型（带残差连接）")
    
    # 加载之前训练的权重作为起点（如果有）
    if os.path.exists("../models/cnn_model_focal.pth"):
        try:
            model.load_state_dict(torch.load("../models/cnn_model_focal.pth", map_location=device))
            print("加载预训练权重进行微调")
        except:
            print("无法加载预训练权重，使用随机初始化")
    
    # 使用焦点损失函数，更加关注难分类样本
    def focal_loss_alpha(outputs, targets, alpha=[1.2, 0.8], gamma=2.0):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        pt = torch.exp(-ce_loss)
        class_weights = torch.tensor(alpha).to(device)[targets]
        focal_loss = class_weights * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    criterion = focal_loss_alpha
    
    # 使用较低的学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.7)

    # 训练参数
    epochs = 30
    best_bus_accuracy = 0.0  # 专门针对bus准确率优化
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    val_bus_accs = []
    val_car_accs = []
    val_accs = []

    print("\n开始针对性训练（重点提升巴士准确率）...")
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_bus_correct = 0
        val_bus_total = 0
        val_car_correct = 0
        val_car_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 分别统计bus和car的准确率
                for i in range(len(labels)):
                    if labels[i] == 0:  # bus
                        val_bus_total += 1
                        if predicted[i] == labels[i]:
                            val_bus_correct += 1
                    else:  # car
                        val_car_total += 1
                        if predicted[i] == labels[i]:
                            val_car_correct += 1

        # 计算指标
        train_loss = train_loss / train_total
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        val_bus_acc = 100 * val_bus_correct / val_bus_total if val_bus_total > 0 else 0
        val_car_acc = 100 * val_car_correct / val_car_total if val_car_total > 0 else 0
        
        # 记录训练历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_bus_accs.append(val_bus_acc)
        val_car_accs.append(val_car_acc)

        # 更新学习率
        scheduler.step()

        # 打印日志
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  训练: Loss={train_loss:.4f}")
        print(f"  验证: Loss={val_loss:.4f}, Total Acc={val_acc:.2f}%")
        print(f"    Bus Acc: {val_bus_acc:.2f}%, Car Acc: {val_car_acc:.2f}%")

        # 保存最佳模型 - 优先考虑bus准确率，但car准确率不能太低
        if val_bus_acc > best_bus_accuracy and val_car_acc >= 92.0:  # 保证car准确率不低于92%
            best_bus_accuracy = val_bus_acc
            torch.save(model.state_dict(), "../models/cnn_model_bus_focus.pth")
            print(f"  保存最佳模型 (Bus Acc: {best_bus_accuracy:.2f}%)")

    # 绘制训练历史
    plot_training_history(val_accs, val_bus_accs + val_car_accs, train_losses, val_losses, save_path="../models/training_history_bus_focus.png")
    
    print(f"\n训练完成, 最佳巴士准确率: {best_bus_accuracy:.2f}%")

if __name__ == "__main__":
    train_focused_on_bus()