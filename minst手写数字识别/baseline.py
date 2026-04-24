import os
import torchvision
from torchvision.datasets import MNIST
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from types import SimpleNamespace


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        # 图片大小为28x28
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # 输入通道数为1，输出通道数为16，卷积核大小为3x3，步长为1，填充为1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 池化层，池化核大小为2x2，步长为2，除以2
        self.fc1 = nn.Linear(
            32 * 7 * 7, 128
        )  # 全连接层，输入特征数为32*7*7，输出特征数为128
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))  # relu激活函数
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()  # 将模型置为训练模式
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # 更新参数
        # if (iter + 1) % 50 == 0:
        #   print(
        #       "Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}".format(
        #           epoch, num_epochs, iter + 1, len(train_dataloader), loss.item()
        #       )
        #   )


def test(model, device, val_dataloader, epoch, num_epochs):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)  # batch中size
            correct += (predicted == labels).sum().item()
        accurary = correct / total
        print("Val_accurary[{}/{}]:{:.4f}".format(epoch, num_epochs, accurary))


if __name__ == "__main__":
    # 检测是否支持mps
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    # 检测是否支持cuda
    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    parameter = SimpleNamespace(
        config=SimpleNamespace(
            model="CNN",
            optim="Adam",
            lr=1e-4,
            batch_size=25,
            num_epochs=10,
            device=device,
        )
    )

    # dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    dataset_root = "/home/xyjiang/miniconda/data_taohy/minst手写数字识别/MNIST/raw"
    dataset = MNIST(dataset_root, train=True, download=True, transform=ToTensor())
    train_data, val_data = utils.data.random_split(dataset, [50000, 10000])
    train_dataloader = utils.data.DataLoader(
        train_data, batch_size=parameter.config.batch_size, shuffle=True
    )
    val_dataloader = utils.data.DataLoader(val_data, batch_size=8, shuffle=False)
    # 模型初始化，选择模型运行设备
    model = CNNnet()
    model.to(torch.device(device))

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=parameter.config.lr)

    for epoch in range(1, parameter.config.num_epochs + 1):
        # 原来这里会用 swanlab.log 记录 epoch，改为打印
        print(f"Epoch {epoch}/{parameter.config.num_epochs}")
        train(
            model,
            device,
            train_dataloader,
            optimizer,
            criterion,
            epoch,
            parameter.config.num_epochs,
        )
        test(model, device, val_dataloader, epoch, parameter.config.num_epochs)

        if not os.path.exists("minst手写数字识别/model_mnist"):
            os.makedirs("minst手写数字识别/model_mnist")
    torch.save(
        model.state_dict(),
        "/home/xyjiang/miniconda/data_taohy/minst手写数字识别/model_mnist/latest_model_mnist.pth",
    )
