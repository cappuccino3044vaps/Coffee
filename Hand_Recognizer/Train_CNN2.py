import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# データ前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_sets = self._load_image_sets()

    def _load_image_sets(self):
        image_sets = []
        for i in range(len(os.listdir(self.image_dir)) // 6):
            image_set = {
                'edge': os.path.join(self.image_dir, f'edges_{i}.png'),
                'feature_0': os.path.join(self.image_dir, f'feature_{i}_0.png'),
                'feature_1': os.path.join(self.image_dir, f'feature_{i}_1.png'),
                'feature_2': os.path.join(self.image_dir, f'feature_{i}_2.png'),
                'feature_3': os.path.join(self.image_dir, f'feature_{i}_3.png'),
                'hand_region': os.path.join(self.image_dir, f'hand_region_{i}.png')
            }
            if all(os.path.exists(path) for path in image_set.values()):
                image_sets.append(image_set)
        return image_sets

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, idx):
        image_set = self.image_sets[idx]
        images = []
        for key in image_set:
            image = Image.open(image_set[key]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        label = 1 if 'positive' in self.image_dir else 0
        return torch.stack(images), label

# データセットとデータローダーの作成
positive_dir = 'dataset2_preprocessed/positive'
negative_dir = 'dataset2_preprocessed/negative'

positive_dataset = CustomDataset(positive_dir, transform=transform)
negative_dataset = CustomDataset(negative_dir, transform=transform)

dataset = positive_dataset + negative_dataset

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# CNNモデル定義
class PalmRecognizer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # EfficientNetを使用
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 最終層を置き換え
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # ArcFace損失用の層
        self.feature_extractor = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()  # シグモイド関数を追加

    def forward(self, x):
        # 各画像を個別に処理
        batch_size, num_images, channels, height, width = x.size()
        # バッチと画像の次元を結合
        x = x.view(batch_size * num_images, channels, height, width)
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        # バッチと画像の次元を分離
        features = features.view(batch_size, num_images, -1)
        # 画像間の平均を取る
        features = features.mean(dim=1)
        return self.sigmoid(self.fc(features))

# 学習ループ
if __name__ == "__main__":
    # 初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmRecognizer(num_classes=2).to(device)
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    # Early Stopping
    best_val_acc = 0
    patience = 5
    counter = 0
    
    # メトリクス記録用
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(30):
        # トレーニングフェーズ
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # バリデーションフェーズ
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学習率スケジューラ更新
        scheduler.step(val_acc)
        
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
                
        # エポックごとの結果を表示
        print(f"Epoch {epoch+1}/{30}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
