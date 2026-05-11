import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import BusCarDataset
from model import CNN, CNNImproved

def evaluate(model_path="../models/cnn_model_bus_focus.pth"):  # 优先使用最新的专注巴士模型
    """
    评估CNN模型在测试集上的性能
    
    Args:
        model_path (str): 模型文件路径
    
    Returns:
        dict: 包含准确率、损失等评估指标
    """
    # 设置设备 - 优先使用GPU，故障降到CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  警告: CUDA不可用，使用CPU评估（速度会比较慢）")
        print(f"使用设备: {device}")

    # 检查模型文件是否存在 - 按优先级查找模型文件
    model_paths = [
        "../models/cnn_model_bus_focus.pth",      # 最新的专注巴士模型
        "../models/cnn_model_enhanced.pth",      # 增强模型
        "../models/cnn_model_balanced.pth",      # 平衡模型
        "../models/cnn_model_focal.pth",         # Focal Loss模型
        "../models/cnn_model.pth"                # 原始模型
    ]
    
    found_model = False
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"使用模型: {model_path}")
            found_model = True
            break
    
    if not found_model:
        print(f"错误: 没有找到任何模型文件！")
        print("请先运行 train.py 训练模型。")
        return None

    # 加载模型权重（临时加载以确定模型类型）
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 尝试根据权重键的特征判断模型类型
    if any('init_conv' in key or 'res_block' in key for key in checkpoint.keys()):
        # 这是改进版模型 (CNNImproved)
        model = CNNImproved(num_classes=2, input_size=64).to(device)
        print("检测到改进版CNN模型（带残差连接）")
    else:
        # 这是基础版模型 (CNN)
        model = CNN(num_classes=2, input_size=64).to(device)
        print("检测到基础版CNN模型")
    
    # 加载模型权重
    model.load_state_dict(checkpoint)
    model.eval()

    # 加载测试数据集（使用测试时的变换）
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_dataset = BusCarDataset("../data/test", transform=test_transform)
    
    if len(test_dataset) == 0:
        print("错误: 测试数据集为空！")
        return None
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"测试集大小: {len(test_dataset)}")

    # 评估
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # 计算每个类别的准确率
    class_correct = [0] * 2
    class_total = [0] * 2

    print("\n开始评估...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            # 计算预测
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 按类别统计
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # 计算指标
    accuracy = 100 * correct / total
    avg_loss = total_loss / total

    # 输出结果
    print("=" * 50)
    print(f"测试集评估结果:")
    print(f"  总样本数: {total}")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  准确率: {accuracy:.2f}%")
    print("\n按类别准确率:")
    
    for i, cls_name in enumerate(BusCarDataset.CLASSES):
        if class_total[i] > 0:
            cls_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {cls_name}: {cls_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {cls_name}: 无测试样本")

    print(f"\n模型来源: {model_path}")

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "class_accuracy": [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                           for i in range(2)],
        "class_names": BusCarDataset.CLASSES
    }


if __name__ == "__main__":
    evaluate()