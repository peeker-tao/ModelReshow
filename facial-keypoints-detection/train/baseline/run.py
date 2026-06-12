import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
import numpy as np
import wandb
from datasets import (
    DatasetLoader,
    TestDataset,
    KeypointCompose,
    RandomHorizontalFlipWithKeypoints,
    RandomAffineWithKeypoints,
    RandomBrightnessContrast,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomCutout,
)
from train import train
from predict import predict, make_submission, make_prediction_table
from output import plot, save_model, load_model
from model import (
    ResNet10Model,
    ResNet50Model,
    EfficientNetB0Model,
    EfficientNetB3Model,
    EfficientNetV2SModel,
    ViTtransformer_B16_pretrained,
    ViTtransformer_L16_pretrained,
)
from utils import compute_mean_std
import pandas as pd
import torchvision.transforms as T

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
STEP_SIZE = 30
GAMMA = 0.1
SPLIT_RATIO = 0.2

run = wandb.init(
    entity="3222703726-huazhong-university-of-science-and-technology",
    project="facial-keypoints-detection",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "ViT_L16_pretrained",
        "dataset": "Facial Keypoints Detection",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "data_augmentation": "RandomHorizontalFlip, RandomAffine, RandomBrightnessContrast, RandomGaussianNoise, RandomGaussianBlur, RandomCutout",
    },
)

train_path = Path("facial-keypoints-detection/data/training.csv")
test_path = Path("facial-keypoints-detection/data/test.csv")
output_csv = Path(
    "facial-keypoints-detection/train/baseline/output/predictions_long.csv"
)
submission_csv = Path("facial-keypoints-detection/train/baseline/output/submission.csv")
lookup_csv = Path("facial-keypoints-detection/data/IdLookupTable.csv")

output_csv.parent.mkdir(parents=True, exist_ok=True)
submission_csv.parent.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
idx = np.arange(len(df_train))
train_idx, val_idx = train_test_split(
    idx, test_size=SPLIT_RATIO, random_state=seed, shuffle=True
)


dataset = DatasetLoader(df_train)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = TestDataset(df_test)

mean_std_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
mean, std = compute_mean_std(mean_std_loader)
print(f"Computed mean: {mean}, std: {std}")
transform_train = T.Compose([T.Normalize(mean=mean, std=std)])
transform_eval = T.Compose([T.Resize((96, 96)), T.Normalize(mean=mean, std=std)])

train_augment = KeypointCompose(
    [
        RandomHorizontalFlipWithKeypoints(p=0.5),
        RandomAffineWithKeypoints(
            degrees=10, translate=0.02, scale=(0.95, 1.05), p=0.6
        ),
        RandomBrightnessContrast(brightness=0.2, contrast=0.2, p=0.5),
        RandomGaussianNoise(std=0.03, p=0.4),
        RandomGaussianBlur(p=0.2, kernel_size=3, sigma=(0.1, 1.0)),
        RandomCutout(p=0.2, size=12, fill=0.0),
    ]
)

dataset = DatasetLoader(df_train, transform=transform_train, augment=train_augment)
test_dataset = TestDataset(df_test, transform=transform_eval)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(DatasetLoader(df_train, transform=transform_eval), val_idx)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
)

model = ViTtransformer_L16_pretrained(num_outputs=30, in_channels=1)
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if num_gpus > 1:
    print(f"Multiple GPUs detected ({num_gpus}) - using DataParallel")
    model = torch.nn.DataParallel(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=STEP_SIZE, gamma=GAMMA
)

scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # 监控loss
    factor=0.5,  # 乘以 0.4（即除以 2.5）
    patience=5,  # 等待 5 个 epoch 不改善才调整
)

num_epochs = EPOCHS
train_losses, val_losses = train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    lr_scheduler=scheduler_base,
    logger=wandb,
)
save_model(model, "facial-keypoints-detection/train/baseline/output/baseline_model.pth")
plot(train_losses, val_losses)
# model = load_model(model, "facial-keypoints-detection/train/baseline/output/baseline_model.pth", device)
predictions = predict(model, test_loader, device)
make_prediction_table(predictions, lookup_csv=lookup_csv, output_csv=output_csv)
make_submission(predictions, lookup_csv=lookup_csv, output_csv=submission_csv)
wandb.finish()
