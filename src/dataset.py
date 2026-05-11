import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BusCarDataset(Dataset):
    
    # 类别名称映射
    CLASSES = ["electric bus", "electric car"]
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = []
        
        # 默认数据增强和预处理
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet均值方差
            ])
        else:
            self.transform = transform

        # 加载数据集
        for label, cls_name in enumerate(self.CLASSES):
            cls_path = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_path):
                img_files = []
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    if os.path.isfile(img_path):
                        # 验证图像文件是否可以打开
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                            # 重新打开图像进行使用
                            img_path = os.path.join(cls_path, img_name)
                            img_files.append(img_path)
                        except Exception as e:
                            print(f"警告: 无法读取图像文件 {img_path}: {e}")
                            continue
                
                for img_path in img_files:
                    self.samples.append((img_path, label))
                
                print(f"类别 '{cls_name}' 加载了 {len(img_files)} 个样本")
            else:
                print(f"警告: 类别目录 {cls_path} 不存在")
        
        if len(self.samples) == 0:
            raise ValueError(f"数据集为空，请检查路径 {root_dir} 下的数据")
        else:
            print(f"总共加载了 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label