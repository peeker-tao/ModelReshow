import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision


def _replace_first_conv(model, in_channels):
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
    )
    nn.init.kaiming_normal_(
        model.features[0][0].weight, mode="fan_out", nonlinearity="relu"
    )
    return model


# 96x96的灰度图像，输入通道数为1,输出30个关键点坐标
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 30)
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # 以kaiming随机初始化权重
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


def resnet10(num_classes=30, in_channels=1):
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


class ResNet10Model(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1):
        super().__init__()
        self.net = resnet10(num_classes=num_outputs, in_channels=in_channels)
        self._init_weights()

    def forward(self, x):
        return self.net(x)

    # 以kaiming随机初始化权重
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class ResNet50Model(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1):
        super().__init__()
        self.net = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        self.net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.net.fc = nn.Linear(self.net.fc.in_features, num_outputs)

    def forward(self, x):
        return self.net(x)

    # 以kaiming随机初始化权重
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class EfficientNetB0Model(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1):
        super().__init__()
        self.net = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
        )
        _replace_first_conv(self.net, in_channels)
        self.net.classifier[1] = nn.Linear(
            self.net.classifier[1].in_features, num_outputs
        )

    def forward(self, x):
        return self.net(x)


class EfficientNetB3Model(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1):
        super().__init__()
        self.net = torchvision.models.efficientnet_b3(
            weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT
        )
        _replace_first_conv(self.net, in_channels)
        self.net.classifier[1] = nn.Linear(
            self.net.classifier[1].in_features, num_outputs
        )

    def forward(self, x):
        return self.net(x)


class EfficientNetB7Model(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1):
        super().__init__()
        self.net = torchvision.models.efficientnet_b7(
            weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT
        )
        _replace_first_conv(self.net, in_channels)
        self.net.classifier[1] = nn.Linear(
            self.net.classifier[1].in_features, num_outputs
        )

    def forward(self, x):
        return self.net(x)


class EfficientNetV2SModel(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1):
        super().__init__()
        self.net = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        )
        _replace_first_conv(self.net, in_channels)
        self.net.classifier[1] = nn.Linear(
            self.net.classifier[1].in_features, num_outputs
        )

    def forward(self, x):
        return self.net(x)


class ViTtransformer_B16_pretrained(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1, image_size=96, patch_size=16):
        super().__init__()
        # 1. 加载并插值预训练权重
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        state_dict = torchvision.models.vision_transformer.interpolate_embeddings(
            image_size=image_size,
            patch_size=patch_size,
            model_state=state_dict,
        )
        if "conv_proj.weight" in state_dict and in_channels == 1:
            state_dict["conv_proj.weight"] = state_dict["conv_proj.weight"].mean(
                dim=1, keepdim=True
            )
        state_dict.pop("heads.head.weight", None)
        state_dict.pop("heads.head.bias", None)

        # 2. 创建默认模型（image_size=224 → 197 个 pos_embed）
        self.net = torchvision.models.vit_b_16(weights=None)

        # 3. 修改 conv_proj 为 1 通道
        self.net.conv_proj = nn.Conv2d(
            in_channels, 768, kernel_size=(16, 16), stride=(16, 16), bias=False
        )

        # 4. 修改 pos_embedding 适配 image_size=96 → (96/16)^2 + 1 = 37
        num_patches = (image_size // patch_size) ** 2
        self.net.encoder.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, 768)
        )

        # 5. 设置 image_size 使 _process_input 断言通过
        self.net.image_size = image_size

        # 6. 加载权重（现在形状完全匹配）
        self.net.load_state_dict(state_dict, strict=False)

        # 7. 替换分类头
        self.net.heads = nn.Sequential(nn.Linear(768, num_outputs))

    def forward(self, x):
        return self.net(x)


class ViTtransformer_L16_pretrained(nn.Module):
    def __init__(self, num_outputs=30, in_channels=1, image_size=96, patch_size=16):
        super().__init__()
        # 1. 加载并插值预训练权重
        weights = torchvision.models.ViT_L_16_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        state_dict = torchvision.models.vision_transformer.interpolate_embeddings(
            image_size=image_size,
            patch_size=patch_size,
            model_state=state_dict,
        )
        if "conv_proj.weight" in state_dict and in_channels == 1:
            state_dict["conv_proj.weight"] = state_dict["conv_proj.weight"].mean(
                dim=1, keepdim=True
            )
        state_dict.pop("heads.head.weight", None)
        state_dict.pop("heads.head.bias", None)

        # 2. 创建默认模型
        self.net = torchvision.models.vit_l_16(weights=None)

        # 3. 修改 conv_proj 为 1 通道（ViT-L 的 embed_dim=1024）
        self.net.conv_proj = nn.Conv2d(
            in_channels, 1024, kernel_size=(16, 16), stride=(16, 16), bias=False
        )

        # 4. 修改 pos_embedding 适配 image_size=96
        num_patches = (image_size // patch_size) ** 2
        self.net.encoder.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, 1024)
        )

        # 5. 设置 image_size 使 _process_input 断言通过
        self.net.image_size = image_size

        # 6. 加载权重
        self.net.load_state_dict(state_dict, strict=False)

        # 7. 替换分类头
        self.net.heads = nn.Sequential(nn.Linear(1024, num_outputs))

    def forward(self, x):
        return self.net(x)


class ViTtransformer(nn.Module):
    def __init__(
        self,
        num_outputs=30,
        in_channels=1,
        image_size=96,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        num_patches = (image_size // patch_size) * (image_size // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_outputs)

        self._init_weights()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(
            self.patch_embed.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
