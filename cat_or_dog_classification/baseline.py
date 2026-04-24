import os
import csv
import torch
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from types import SimpleNamespace
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetLoader(Dataset):
    def __init__(self, csv_path):
        self.csv_file = csv_path
        with open(self.csv_file, "r") as f:
            self.data = list(csv.reader(f))

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def get_image(self, image_path):
        img_path = os.path.join(self.current_dir, "data", image_path)
        image = Image.open(img_path).convert("RGB")
        image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return image_transform(image)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = self.get_image(image_path)
        return image, int(label)

    def __len__(self):
        return len(self.data)


def train(model, device, train_loader, optimizer, criterion, epoch, num_epochs):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # if batch_idx % 50 == 0:
        print(
            "Train Epoch: {}/{} [{}/{}]\tLoss: {:.4f}".format(
                epoch,
                num_epochs,
                batch_idx,
                len(train_loader),
                loss.item(),
            )
        )


def test(model, device, val_loader, epoch, num_epochs):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = correct / total
    print("Val Accuracy: {}/{}: {:.4f}".format(epoch, num_epochs, accuracy))


if __name__ == "__main__":

    train_dataset = DatasetLoader(
        "/home/xyjiang/miniconda/data_taohy/cat_or_dog_classification/data/train.csv"
    )
    val_dataset = DatasetLoader(
        "/home/xyjiang/miniconda/data_taohy/cat_or_dog_classification/data/val.csv"
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features  # 获取 ResNet50 最后全连接层的输入特征数
    model.fc = torch.nn.Linear(in_features, 2)  # 替换最后的全连接层，输出类别数为2

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(torch.device(device))  # 将模型移动到指定设备

    parameter = SimpleNamespace(
        model="ResNet50",
        optim="Adam",
        lr=1e-4,
        batch_size=10,
        num_epochs=10,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameter.lr)

    print("训练集：{:.4f},验证集：{:.4f}".format(len(train_dataset), len(val_dataset)))
    for epoch in range(1, parameter.num_epochs + 1):
        train(
            model,
            parameter.device,
            train_loader,
            optimizer,
            criterion,
            epoch,
            parameter.num_epochs,
        )
        test(model, parameter.device, val_loader, epoch, parameter.num_epochs)

    if not os.path.exists("cat_or_dog_classification/model_resnet50"):
        os.makedirs("cat_or_dog_classification/model_resnet50")
    torch.save(
        model.state_dict(),
        "/home/xyjiang/miniconda/data_taohy/cat_or_dog_classification/model_resnet50/model_resnet50.pth",
    )
