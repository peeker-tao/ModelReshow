import os
import random
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
from torch import nn, optim, utils
from baseline import CNNnet


def test(device, val_loader_or_dataset, num_samples=50, out_dir="predictions"):
    """
    从验证集随机抽取 num_samples 张图片，用 model 在 device 上预测，
    将带有预测结果和真实标签的图片保存到 out_dir，并打印准确率。

    参数:
    - model: 已加载权重的 PyTorch 模型
    - device: 'cpu' / 'cuda' / 'mps'
    - val_loader_or_dataset: 验证集的 DataLoader 或 Dataset（支持 Subset）
    - num_samples: 要抽取的样本数，默认50
    - out_dir: 保存带注释图片的目录
    """
    # 支持传入 DataLoader 或 Dataset
    if isinstance(val_loader_or_dataset, torch.utils.data.DataLoader):
        dataset = val_loader_or_dataset.dataset
    else:
        dataset = val_loader_or_dataset
    try:
        total = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        if hasattr(dataset, "data"):
            total = len(dataset.data)  # type: ignore[union-attr]
        else:
            raise RuntimeError(
                "无法确定验证集长度，请传入支持 __len__ 的 Dataset 或 DataLoader"
            )

    if total == 0:
        print("验证集为空，无法执行测试")
        return

    num = min(num_samples, total)
    indices = random.sample(range(total), num)

    model = CNNnet()  # 创建模型对象
    model.load_state_dict(
        torch.load(
            "/home/xyjiang/miniconda/data_taohy/minst手写数字识别/model_mnist/latest_model_mnist.pth"
        )
    )  # 加载参数字典
    model.to(device)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    images = []
    trues = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        trues.append(int(label))

    batch = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(batch)
        preds = outputs.argmax(dim=1).cpu().tolist()

    for i, idx in enumerate(indices):
        img = images[i]
        true = trues[i]
        pred = preds[i]

        pil = to_pil_image(img.cpu())
        draw = ImageDraw.Draw(pil)
        text = f"pred: {pred}   label: {true}"
        # 在左上角写上预测与真实标签（灰度图像用白色）
        draw.text((2, 2), text, fill=255)
        fname = f"{idx}_pred{pred}_label{true}.png"
        pil.save(os.path.join(out_dir, fname))

    print(f"Saved {num} images to '{out_dir}'. ")


if __name__ == "__main__":
    print("请在 baseline.py 中训练好模型后，调用 test() 函数来查看预测结果")
    # 检测是否支持cuda
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_data, val_data = utils.data.random_split(dataset, [50000, 10000])
    test(device, val_data)
