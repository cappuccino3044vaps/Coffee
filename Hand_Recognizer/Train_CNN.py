import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision.models import  EfficientNet_B0_Weights
from torchvision import transforms
import os

# データ前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # パースペクティブ変換追加
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ランダムシフト追加
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

positive_transform = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.8),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3))
])

# データセットクラスを修正
class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, positive_transform=None):
        self.dataset = dataset
        self.positive_transform = positive_transform
        self.targets = dataset.targets
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if label == 1 and self.positive_transform:  # Positiveクラスのみ拡張
            image = self.positive_transform(image)
        if label == 0:  # Negativeクラス用の拡張
            image = transforms.RandomChoice([
                transforms.RandomRotation(180),
                transforms.RandomPerspective(0.8),
                transforms.ColorJitter(brightness=0.5),
            ])(image)
        return image, label

def load_dataset(dataset_path, img_size=(224, 224)):
    X, y = [], []
    for label, category in enumerate(["negative", "positive"]):
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.resize(image, img_size)
                image = image / 255.0
                X.append(image)
                y.append(label)
    return np.array(X), np.array(y)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image_np = (self.images[idx]*255).astype('uint8')
        # ...existing code...
        pil_img = Image.fromarray(image_np)
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, self.labels[idx]

# 既存のデータ読み込み部分をカスタムDatasetに置き換え
# dataset = datasets.ImageFolder(root='dataset', transform=transform)
X, y = load_dataset('dataset', img_size=(224,224))
dataset_all = CustomDataset(X, y, transform=transform)

train_size = int(0.8 * len(dataset_all))
val_size = len(dataset_all) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset_all, [train_size, val_size])

# データローダー作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# CNNモデル定義
class PalmRecognizer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # EfficientNetを使用
        self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
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
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return self.sigmoid(self.fc(features))

# Focal Lossの導入
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
# 手のひらセグメンテーション関数を追加
def extract_palm_region(image):
    # OpenCVを使用して手のひら領域を抽出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return cv2.bitwise_and(image, image, mask=mask)

def calculate_fpr(preds, targets):
    fp = ((preds == 1) & (targets == 0)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()
    return fp / (fp + tn)
# データローダー作成(セグメンテーション適用)
class PalmSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # PIL画像をnumpy配列に変換してセグメンテーション
        image_np = np.array(image)
        segmented = extract_palm_region(image_np)
        # numpy配列をPIL画像に戻す
        image = Image.fromarray(segmented)
        return image, label

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # 正規化
        logits = torch.nn.functional.normalize(logits, p=2, dim=1)
        
        # コサイン類似度を計算
        cosine = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        
        # ターゲットロジットを計算
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        target_logits = torch.cos(theta + self.m)
        
        # 最終的なロジットを計算
        logits = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        
        return self.ce(logits, labels)

train_dataset = PalmSegmentationDataset(train_dataset)
val_dataset = PalmSegmentationDataset(val_dataset)
# 学習ループ
if __name__ == "__main__":
    # 初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmRecognizer(num_classes=2).to(device)
    
    # 損失関数とオプティマイザ
    criterion = ArcFaceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    class_weights = torch.tensor([10.0, 1.0]).to(device)  # Negative:Positive = 2:1
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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