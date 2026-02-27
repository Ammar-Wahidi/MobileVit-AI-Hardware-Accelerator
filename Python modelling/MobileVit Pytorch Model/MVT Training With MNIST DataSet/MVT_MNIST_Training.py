import os
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from torchsummary import summary

# PyTorch Implementation
class MV2(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, expansion_ratio=1):
        super().__init__()
        hidden_dim = out_channels * expansion_ratio
        self.use_residual = strides == 1 and in_channels == out_channels

        # 1. Point-wise conv (expand)
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 2. Depth-wise conv
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=strides, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 3. Point-wise linear conv (project)
        self.pw_project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        skip = x
        x = self.pw_expand(x)
        x = self.dw_conv(x)
        x = self.pw_project(x)
        if self.use_residual:
            x = x + skip
        return x


class LocalRepresentation(nn.Module):
    def __init__(self, in_channels, num_filters, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(num_filters, out_dim, kernel_size=1, stride=1, padding= 0 )
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_dim = mlp_dim
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        skip = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = skip + attn_out

        skip2 = x
        x_norm = self.norm2(x)

        x = self.fc1(x_norm)
        x = self.act(x)

        y1 = self.fc2(x)

        x = skip2 + y1

        return x

class Fusion(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, x_fusion):
        x_f = self.conv1(x_fusion)
        x_f = self.bn1(x_f)
        x_f = self.act1(x_f)

        x_concat = torch.cat([x, x_f], dim=1)

        x_out = self.conv2(x_concat)
        x_out = self.bn2(x_out)
        x_out = self.act2(x_out)

        return x_out

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, num_filters, dim, patch_size=2, num_layers=1):
        super().__init__()
        self.local_rep = LocalRepresentation(in_channels, num_filters, dim)
        self.transformers = nn.ModuleList([TransformerEncoder(dim, num_heads=4, mlp_dim=dim*2) for _ in range(num_layers)])
        self.fusion = Fusion(in_channels, dim)
        self.patch_size = patch_size

    def forward(self, x):
        b, C, H, W = x.shape
        x_local = self.local_rep(x)


        # Patch embedding for transformer
        ph = pw = self.patch_size
        _, _, h, w = x_local.shape
        x_patches = rearrange(x_local, 'b d (h ph) (w pw) -> b (h w ph pw) d', ph=ph, pw=pw)

        # Transformer layers
        for transformer in self.transformers:
            x_patches = transformer(x_patches)

        # Reshape back
        x_global = rearrange(x_patches, 'b (h w ph pw) d -> b d (h ph) (w pw)',
                             h=h//ph, w=w//pw, ph=ph, pw=pw)

        # Fusion
        out = self.fusion(x, x_global)
        return out

class MobileViT_XXS(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        # Stage 1
        self.mv2_1 = MV2(16, 16)

        # Stage 2
        self.mv2_2a = MV2(16, 16, strides=2)
        self.mv2_2b = MV2(16, 24)
        self.mv2_2c = MV2(24, 24)

        # Stage 3
        self.mv2_3a = MV2(24, 48, strides=2)
        self.mvit_3b = MobileViTBlock(48, 48, 64, patch_size=2, num_layers=2)

        # Stage 4
        self.mv2_4a = MV2(48, 64, strides=2)
        self.mvit_4b = MobileViTBlock(64, 64, 80, patch_size=2, num_layers=4)

        # Stage 5
        self.mv2_5a = MV2(64, 80, strides=2)
        self.mvit_5b = MobileViTBlock(80, 80, 96, patch_size=2, num_layers=3)

        # Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(80, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(320),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.stem(x)


        x = self.mv2_1(x)


        x = self.mv2_2a(x)
        x = self.mv2_2b(x)
        x = self.mv2_2c(x)

        x = self.mv2_3a(x)

        x = self.mvit_3b(x)

        x = self.mv2_4a(x)
        x = self.mvit_4b(x)

        x = self.mv2_5a(x)
        x = self.mvit_5b(x)

        x = self.head_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

model = MobileViT_XXS(in_channels=3, num_classes=10)
x = torch.randn(2, 3, 256, 256)  # batch of 2 images
y = model(x)
print(y.shape)  # [2, 10]


######################################################################################################################################
# ===== Dataset & Dataloaders =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # resize MNIST (28x28) -> 256x256
    transforms.Grayscale(num_output_channels=3),  # convert 1-channel to 3-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize
])

mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# 70% train, 30% validation
train_size = int(0.7 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size
train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ===== Model, Loss, Optimizer =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileViT_XXS(in_channels=3, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== Model Summary =====
print("Model Summary:")
summary(model, (3, 256, 256))

# ===== Training & Evaluation =====
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ===== Training Loop =====
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

for name, param in model.named_parameters():
    print(name, param.shape)

for name, param in model.named_parameters():
    print(f"{name} -> {param.view(-1)[:5]}")

torch.save(model.state_dict(), "mobilevit_weights.pth")
model.load_state_dict(torch.load("mobilevit_weights.pth"))

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

# Save all parameter names, dtypes, shapes, and values into a text file
with open("all_parameters.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(
            f"{name} | dtype: {param.dtype} | shape: {tuple(param.shape)} "
            f"-> {param.view(-1).tolist()}\n"
        )


