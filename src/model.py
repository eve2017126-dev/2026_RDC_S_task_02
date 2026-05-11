import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes=2, input_size=64):
        super(CNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes

        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积：3通道输入 -> 16通道输出
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积：16通道 -> 32通道
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层卷积：32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算特征图大小（经过3次MaxPool2d，每次缩小2倍）
        feature_size = input_size // (2 ** 3)  # 64 // 8 = 8
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),  # Dropout防止过拟合
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNImproved(nn.Module):
 
    def __init__(self, num_classes=2, input_size=64):
        super(CNNImproved, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 残差块1
        self.res_block1 = self._make_residual_block(32, 64)
        # 残差块2
        self.res_block2 = self._make_residual_block(64, 128)
        # 残差块3
        self.res_block3 = self._make_residual_block(128, 256)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x