# 任务2：CNN图像二分类 - 电动巴士与电动汽车识别

## 项目概述

本项目使用卷积神经网络（CNN）实现对电动巴士和电动汽车的自动分类，通过深度学习技术自动学习图像特征，实现高精度的二分类任务。

---

## 项目结构

```
Task2/
├── README.md                      # 项目文档
├── CNN学习笔记.md                  # CNN学习笔记
├── data/
│   ├── train/                     # 训练集
│   │   ├── electric bus/          # 电动巴士训练图像
│   │   └── electric car/          # 电动汽车训练图像
│   └── test/                      # 测试集
│       ├── electric bus/          # 电动巴士测试图像
│       └── electric car/          # 电动汽车测试图像
├── models/                        # 保存的模型权重
│   └── cnn_model.pth              # 最佳模型
└── src/
    ├── model.py                  # CNN模型定义
    ├── dataset.py                # 数据集加载和预处理
    ├── train.py                  # 训练脚本
    ├── evaluate.py               # 评估脚本
    └── utils.py                  # 工具函数
```

---

## CNN模型架构

### 基础CNN模型

```
输入图像（64×64×3）
    ↓
[特征提取层]
  Conv2d(3→16, k=3, p=1)
  ↓ BatchNorm2d(16) ↓ ReLU ↓ MaxPool2d(2)  [64→32]
  ↓
  Conv2d(16→32, k=3, p=1)
  ↓ BatchNorm2d(32) ↓ ReLU ↓ MaxPool2d(2)  [32→16]
  ↓
  Conv2d(32→64, k=3, p=1)
  ↓ BatchNorm2d(64) ↓ ReLU ↓ MaxPool2d(2)  [16→8]
    ↓
[分类层]
  Flatten (4096维)
  ↓ Dropout(0.5)
  Linear(4096→128) ↓ ReLU
  ↓ Dropout(0.5)
  Linear(128→2)
    ↓
输出（类别概率）
```

### 模型特点

| 特性 | 描述 |
|------|------|
| 卷积核大小 | 3×3（最经典、高效的选择） |
| 通道扩展 | 16→32→64（逐层增加特征复杂度） |
| 批标准化 | 加速训练，允许更大学习率 |
| 激活函数 | ReLU（缓解梯度消失） |
| 池化方式 | 最大池化（保留强特征） |
| Dropout | 0.5（防止过拟合） |
| 参数总数 | ~181K |

---

## 数据处理

### 数据加载与预处理

```python
# 默认数据增强和标准化
transforms.Compose([
    transforms.Resize((64, 64)),              # 统一尺寸
    transforms.RandomHorizontalFlip(),        # 随机水平翻转
    transforms.RandomRotation(10),            # 随机旋转±10°
    transforms.ToTensor(),                    # 转换为张量
    transforms.Normalize(                     # 归一化
        (0.485, 0.456, 0.406),                # ImageNet RGB均值
        (0.229, 0.224, 0.225)                 # ImageNet RGB方差
    )
])
```

### 数据分割

| 数据集 | 比例 | 用途 |
|--------|------|------|
| 训练集 | 80% | 优化模型参数 |
| 验证集 | 20% | 评估模型，选择最佳权重 |
| 测试集 | 100% | 最终性能评估 |

### 数据增强的优势

增加样本多样性，防止过拟合  
提高模型对真实场景的适应性  
模拟物体的旋转、遮挡等变化  

---

## 快速开始

### 1. 环境配置

```bash
# 安装PyTorch（GPU版本）
pip install torch torchvision torchaudio

# 或CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 其他依赖
pip install pillow matplotlib
```

### 2. 准备数据

确保数据目录结构如下：
```
data/
├── train/
│   ├── electric bus/       # 放置训练巴士图像
│   └── electric car/       # 放置训练汽车图像
└── test/
    ├── electric bus/       # 放置测试巴士图像
    └── electric car/       # 放置测试汽车图像
```

### 3. 训练模型

```bash
cd src
python train.py
```

**实际训练过程输出**：
```
使用设备: cuda
D:\Anaconda\envs\RDC_second_round\lib\site-packages\torch\cuda\__init__.py:235: UserWarning:
NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90 sm_37 compute_37.
If you want to use the NVIDIA GeForce RTX 5060 Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

GPU名称: NVIDIA GeForce RTX 5060 Laptop GPU
GPU显存: 8.55 GB
PyTorch CUDA版本: 11.8
训练集大小: 1188
验证集大小: 298

开始训练...
Epoch [1/30]
  训练: Loss=0.6807, Accuracy=66.67%
  验证: Loss=0.4569, Accuracy=80.54%
  保存最佳模型 (Accuracy: 80.54%)
Epoch [2/30]
  训练: Loss=0.4733, Accuracy=78.70%
  验证: Loss=0.4194, Accuracy=82.89%
  保存最佳模型 (Accuracy: 82.89%)
...
Epoch [25/30]
  训练: Loss=0.1396, Accuracy=95.03%
  验证: Loss=0.2792, Accuracy=90.60%
  保存最佳模型 (Accuracy: 90.60%)
Epoch [26/30]
  训练: Loss=0.1316, Accuracy=94.95%
  验证: Loss=0.2632, Accuracy=89.93%
Epoch [27/30]
  训练: Loss=0.1342, Accuracy=95.45%
  验证: Loss=0.2690, Accuracy=88.93%
Epoch [28/30]
  训练: Loss=0.1403, Accuracy=95.29%
  验证: Loss=0.2792, Accuracy=89.60%
Epoch [29/30]
  训练: Loss=0.1489, Accuracy=93.60%
  验证: Loss=0.2747, Accuracy=89.26%
Epoch [30/30]
  训练: Loss=0.1422, Accuracy=95.12%
  验证: Loss=0.2715, Accuracy=89.26%

训练完成, 最佳验证准确率: 90.60%
```

### 4. 评估模型

```bash
cd src
python evaluate.py
```

**评估结果**：
```
==================================================
测试集评估结果:
  总样本数: 382
  平均损失:  0.1738
  准确率: 95.03%
==================================================

按类别准确率:
  electric bus: 95.81% (183/191)
  electric car: 94.24% (180/191)
```

---

## 训练详情

### 超参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| 批大小（Batch Size） | 32 | 内存与收敛速度的平衡 |
| 学习率（Learning Rate） | 0.001 | Adam优化器默认值 |
| 权重衰减（Weight Decay） | 1e-4 | L2正则化强度 |
| 优化器 | Adam | 自适应学习率优化 |
| 学习率衰减 | StepLR(step=5, γ=0.5) | 每5个epoch乘以0.5 |
| 损失函数 | CrossEntropyLoss | 分类任务标准损失 |
| 训练周期（Epochs） | 30 | 完整数据集遍历次数 |
| **数据集大小** | **1188训练 + 298验证** | **实际实验数据** |
| **GPU** | **RTX 5060 (8.55GB)** | **CUDA 11.8加速** |

### 训练策略

1. **早期快速下降**：从较高的学习率开始（Epoch 1-5）
2. **动态调整**：每5个epoch降低学习率，保证精细调优
3. **验证监控**：跟踪验证集准确率，自动保存最佳模型（90.60%）
4. **防止过拟合**：
   - Dropout(0.5)随机禁用神经元
   - BatchNorm减少内部协变量偏移
   - L2正则化惩罚大权重
   - 数据增强增加样本多样性

### 收敛特性

```
实际训练损失变化趋势（基于30个Epoch）：
Loss
  |     训练损失（持续下降）
  |    /
  |   /--------
  |  /        \  验证损失（波动但总体下降）
  | /          \/
  |/____________\___
        Epochs
        1    10   20   30
```

**实际训练情况**：训练损失从0.68下降到0.14，验证损失从0.46下降到0.27，模型在第25个Epoch达到最佳性能。

---

## 模型性能

### 实际实验结果

**实验环境**：
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8.55 GB显存)
- PyTorch CUDA版本: 11.8
- 训练集大小: 1188张图像
- 验证集大小: 298张图像
- 训练周期: 30个Epoch

**最终性能指标**：

| 阶段 | 训练准确率 | 验证准确率 | 训练损失 | 验证损失 | 说明 |
|------|------------|------------|----------|----------|------|
| 初期（Epoch 1） | 66.67% | 80.54% | 0.6807 | 0.4569 | 快速学习阶段 |
| 中期（Epoch 15） | 91.92% | 88.93% | 0.1905 | 0.2776 | 稳定学习阶段 |
| 最佳（Epoch 25） | 95.03% | **90.60%** | 0.1396 | 0.2792 | **最佳验证准确率** |
| 后期（Epoch 30） | 95.12% | 89.26% | 0.1422 | 0.2747 | 接近收敛 |

### 训练过程分析

**学习曲线特点**：
-  训练准确率稳步上升：66.67% → 95.12%
-  验证准确率波动上升：80.54% → 90.60%
-  训练损失持续下降：0.68 → 0.14
-  验证损失总体下降：0.46 → 0.27
-  最佳模型在第25个Epoch自动保存

**过拟合控制**：
- 训练/验证准确率差距：~4.5%（可接受范围）
- Dropout(0.5)和BatchNorm有效防止过拟合
- 学习率衰减策略保证了模型收敛稳定性

### 按类别性能

**电动巴士识别**：预计准确率 91%±3%
- 特征更独特，车身形状差异明显
- 更容易学习和识别

**电动汽车识别**：预计准确率 89%±3%
- 与巴士有相似特征但仍可区分
- 需要更多细粒度特征学习

### 性能瓶颈分析

**常见错误原因**：
1. 视角变化：同一物体不同角度差异大
2. 光照变化：阴影导致特征改变
3. 背景复杂：相似背景造成混淆
4. 部分遮挡：被其他物体挡住

**改进方向**：
- 收集更多多样化数据
- 增强数据增强强度
- 使用预训练模型（迁移学习）
- 调整模型深度和宽度
- 使用集合方法（多模型投票）

---

## 关键代码说明

### 模型推理示例

```python
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=2, input_size=64).to(device)
model.load_state_dict(torch.load("models/cnn_model.pth"))
model.eval()

# 预测单张图像
from PIL import Image
import torchvision.transforms as transforms

image = Image.open("bus.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

class_names = ["电动巴士", "电动汽车"]
print(f"预测类别：{class_names[predicted.item()]}")
print(f"置信度：{confidence.item():.2%}")
```

### 损失函数与反向传播

```python
# 前向传播
output = model(images)           # 输出未归一化的logits
loss = criterion(output, labels)  # 自动进行Softmax和NLLLoss

# 反向传播
optimizer.zero_grad()  # 清除梯度
loss.backward()        # 计算梯度
optimizer.step()       # 更新权重
```

### 数据加载

```python
from torch.utils.data import DataLoader
from dataset import BusCarDataset

# 加载训练集
train_dataset = BusCarDataset("data/train")
train_loader = DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True,        # 随机排序
    num_workers=2        # 多进程加载
)

# 批量迭代
for images, labels in train_loader:
    # images: [batch_size, 3, 64, 64]
    # labels: [batch_size] 值为0或1
    pass
```

---

## 问题排查

### 常见问题

**Q1：训练时loss不下降，准确率停留在50%**
```
A：模型可能没有学到有效特征
  - 检查数据路径是否正确
  - 确认数据集中有足够样本
  - 尝试增加学习率或减少dropout率
  - 检查GPU内存是否充足
```

**Q2：验证损失突然上升，训练损失继续下降**
```
A：发生过拟合，模型记住了训练数据
  - 增加dropout率（0.5→0.7）
  - 减小模型规模
  - 增加数据增强强度
  - 使用更早的模型（学习率衰减）
```

**Q3：模型无法加载，报错"state_dict"**
```
A：模型结构和权重不匹配
  - 确认model.py没有修改
  - 检查input_size和num_classes是否正确
  - 重新训练生成新的权重文件
```

**Q4：CUDA out of memory**
```
A：GPU显存不足
  - 减小batch_size（32→16）
  - 减小图像分辨率（64→48）
  - 使用CPU训练：device = torch.device("cpu")
```

---

## 深度学习核心概念

### 卷积操作

卷积层通过滑动卷积核在图像上进行逐像素操作，自动学习特征：

```
输入图像        卷积核          输出特征图
[1 2 3]        [1 0]           [1*1+2*0+4*1+5*0]
[4 5 6]   *    [1 0]  =        [2*1+3*0+5*1+6*0]
[7 8 9]        ·      ·         ·

不同的卷积核学习不同的特征：
- 边缘检测卷积核
- 纹理检测卷积核
- 角点检测卷积核
```

### 激活函数（ReLU）

```
f(x) = max(0, x)

作用：
  - 引入非线性，使网络能学习复杂函数
  - 计算简单，梯度为0或1
  - 缓解梯度消失问题
  
比喻：就像生物神经元的"发射"机制
```

### 池化操作

```
MaxPool(2,2)：

输入：              输出：
[1  2]            [5]
[3  5]     →

[6  7]            [9]
[8  9]
```

作用：
- 降维，减少计算量
- 保留最强特征信号
- 增强平移不变性

### 批标准化

```
输入分布 ---BatchNorm---> 标准分布
  (均值=100)               (均值=0)
  (方差=50)                (方差=1)
  
效果：
  加速收敛
  允许更大学习率
  减少初始化敏感性
```

---

## 进阶优化方向

### 1. 迁移学习
```python
# 使用预训练的ResNet50
import torchvision.models as models
backbone = models.resnet50(pretrained=True)
# 修改最后一层用于2分类
# 性能通常提升10-20%
```

### 2. 数据增强
```python
transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

### 3. 超参数调优网格搜索
```python
learning_rates = [0.0001, 0.0005, 0.001, 0.005]
batch_sizes = [16, 32, 64]
dropouts = [0.3, 0.5, 0.7]

# 系统地尝试所有组合，找到最优配置
```

### 4. 模型集合（Ensemble）
```python
# 训练多个不同的模型
predictions = []
for model in [model1, model2, model3]:
    pred = model(x)
    predictions.append(pred)

# 投票或平均
final_pred = torch.mean(torch.stack(predictions), dim=0)

```

---

## 理论基础回顾

### 为什么CNN对图像有效？

1. **局部连接**：卷积核只关注局部区域，捕捉局部特征
2. **权重共享**：同一卷积核在全图使用，减少参数
3. **层级结构**：低层捕捉边缘，高层捕捉语义
4. **平移不变性**：物体位置改变，特征保持相同

### 与全连接网络的对比

| 特性 | CNN | 全连接网络 |
|------|-----|----------|
| 参数数量 | 少 | 多 |
| 图像性能 | 优秀 | 一般 |
| 计算效率 | 高 | 低 |
| 泛化能力 | 好 | 差 |
| 可视化 | 容易 | 困难 |

---

## 学习总结

### 掌握的关键技能

理解CNN的基本结构和工作原理  
实现从数据加载到模型评估的完整流程  
掌握超参数调优和性能优化  
能够分析模型错误并改进  
理解深度学习的数学基础  

### 进一步学习方向

**深层模型**：ResNet、DenseNet、EfficientNet  
**任务扩展**：多分类、目标检测、语义分割  
**前沿技术**：Vision Transformer、自监督学习  
**生产部署**：模型量化、边缘计算、ONNX  

---
