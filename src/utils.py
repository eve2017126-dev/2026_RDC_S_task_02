import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

def count_dataset_samples(root_dir):

    classes = ["electric bus", "electric car"]
    counts = {}
    
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        if os.path.exists(cls_path):
            counts[cls_name] = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        else:
            counts[cls_name] = 0
    
    return counts


def plot_training_history(train_accs, val_accs, train_losses, val_losses, save_path=None):

    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练和验证损失")
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="训练准确率")
    plt.plot(val_accs, label="验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("训练和验证准确率")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"训练历史图已保存到 {save_path}")
    else:
        plt.show()


def visualize_images(dataset, num_samples=5, save_path=None):

    plt.figure(figsize=(15, 5))
    
    for i in range(min(num_samples, len(dataset))):
        image, label = dataset[i]
        # 将Tensor转换回PIL图像
        image = transforms.ToPILImage()(image)
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"类别: {dataset.CLASSES[label]}")
        plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"可视化图像已保存到 {save_path}")
    else:
        plt.show()


def predict_single_image(image_path, model, transform, device):

    model.eval()
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 预处理
    image = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()


def print_model_summary(model, input_size=(3, 64, 64)):

    print("=" * 60)
    print("模型结构摘要")
    print("=" * 60)
    
    # 打印模型
    print(model)
    
    print("\n" + "=" * 60)
    print("参数统计")
    print("=" * 60)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试前向传播
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n输入形状: {input_size}")
    print(f"输出形状: {output.shape}")


if __name__ == "__main__":
    # 示例用法
    print("数据集统计:")
    train_counts = count_dataset_samples("../data/train")
    test_counts = count_dataset_samples("../data/test")
    
    print(f"训练集: {train_counts}")
    print(f"测试集: {test_counts}")