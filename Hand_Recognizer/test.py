import torch
import torch.nn as nn
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
        return torch.stack(images)

# データセットとデータローダーの作成
image_dir = 'onishi_preprocessed'  # ここに入力フォルダのパスを追加
dataset = CustomDataset(image_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

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

# モデルの評価
def classify_images(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            results.append(predicted.item())
    return results

if __name__ == "__main__":
    # 初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmRecognizer(num_classes=2).to(device)
    
    # 学習済みモデルの読み込み
    model.load_state_dict(torch.load("best_model.pth"))
    
    # 画像の分類
    results = classify_images(model, data_loader, device)
    
    # 結果を表示
    for i, result in enumerate(results):
        label = "positive" if result == 1 else "negative"
        print(f"Image {i}: {label}")
