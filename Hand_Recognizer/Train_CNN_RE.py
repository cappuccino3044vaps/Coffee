import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Subset

class Config:
    """ハイパーパラメータと設定値を管理するクラス"""
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 5
    CLASS_WEIGHTS = [10.0, 1.0]  # [Negative, Positive]
    DATA_ROOT = 'dataset'
    MODEL_SAVE_PATH = 'best_model.pth'

class ImageTransforms:
    """画像変換処理を管理するクラス"""
    @staticmethod
    def get_base_transform():
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_positive_transform():
        return transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.3, 0.3))
        ])

class LossFunctions:
    """損失関数を管理するクラス"""
    @staticmethod
    def get_focal_loss(alpha=0.75, gamma=2.0):
        """Focal Lossを取得"""
        class FocalLoss(nn.Module):
            def __init__(self, alpha, gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
                pt = torch.exp(-BCE_loss)
                F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
                return F_loss.mean()
        return FocalLoss(alpha, gamma)

    @staticmethod
    def get_arcface_loss(s=30.0, m=0.5):
        """ArcFace Lossを取得"""
        class ArcFaceLoss(nn.Module):
            def __init__(self, s, m):
                super().__init__()
                self.s = s
                self.m = m
                self.ce = nn.CrossEntropyLoss()

            def forward(self, logits, labels):
                logits = torch.nn.functional.normalize(logits, p=2, dim=1)
                cosine = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)
                theta = torch.acos(cosine)
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
                target_logits = torch.cos(theta + self.m)
                logits = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
                return self.ce(logits, labels)
        return ArcFaceLoss(s, m)

class PalmSegmentation:
    """手のひらセグメンテーション関連の処理"""
    @staticmethod
    def extract_palm_region(image):
        """手のひら領域を抽出"""
        # RGB -> BGR に変換してからHSVに
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        return cv2.bitwise_and(image, image, mask=mask)

class Metrics:
    """評価指標計算用クラス"""
    @staticmethod
    def calculate_fpr(preds, targets):
        """False Positive Rateを計算"""
        fp = ((preds == 1) & (targets == 0)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()
        return fp / (fp + tn)
    
class BalancedDataset(torch.utils.data.Dataset):
    """クラス不均衡を補正するデータセットクラス"""
    def __init__(self, dataset, positive_transform=None):
        self.dataset = dataset
        self.positive_transform = positive_transform
        if isinstance(dataset, Subset):
            self.targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            self.targets = dataset.targets
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 既にテンソルの場合は PIL 画像に変換
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        if label == 1 and self.positive_transform:
            image = self.positive_transform(image)
        if label == 0:
            # Convert PIL -> Tensor, apply RandomChoice, then Tensor -> PIL
            tensor_img = transforms.ToTensor()(image)
            tensor_img = transforms.RandomChoice([
                transforms.RandomRotation(180),
                transforms.RandomPerspective(0.8),
                transforms.ColorJitter(brightness=0.5),
            ])(tensor_img)
            image = transforms.ToPILImage()(tensor_img)
        return image, label

class PalmSegmentationDataset(torch.utils.data.Dataset):
    """セグメンテーションを適用したデータセット"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # RGBモードへ変換
        image = image.convert('RGB')
        image_np = np.array(image)
        segmented = PalmSegmentation.extract_palm_region(image_np)
        image = Image.fromarray(segmented)
        return image, label
    
class PalmRecognizer(nn.Module):
    """手のひら認識モデル"""
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
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

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return self.fc(features)

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
        pil_img = Image.fromarray(image_np)
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, self.labels[idx]

class ModelTrainer:
    """モデルの訓練を管理するクラス"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PalmRecognizer(num_classes=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(Config.CLASS_WEIGHTS).to(self.device)
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', patience=3, factor=0.5
        )
        self.best_val_acc = 0
        self.patience_counter = 0
        self.loss_fn = LossFunctions.get_arcface_loss()  # デフォルトでArcFace Lossを使用
        self.metrics = Metrics()

    def prepare_data(self):
        """データの準備と前処理"""
        base_transform = ImageTransforms.get_base_transform()

        # ImageFolderを使用する箇所はコメントアウト
        # dataset = datasets.ImageFolder(root=Config.DATA_ROOT, transform=base_transform)
        # train_size = int(0.8 * len(dataset))
        # val_size = len(dataset) - train_size
        # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        # load_dataset + CustomDataset を使用
        X, y = load_dataset(Config.DATA_ROOT, (Config.IMG_SIZE, Config.IMG_SIZE))
        dataset_all = CustomDataset(X, y, transform=base_transform)
        train_size = int(0.8 * len(dataset_all))
        val_size = len(dataset_all) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset_all, [train_size, val_size])

        train_set = BalancedDataset(train_set, ImageTransforms.get_positive_transform())
        val_set = BalancedDataset(val_set)

        train_set = PalmSegmentationDataset(train_set)
        val_set = PalmSegmentationDataset(val_set)

        self.train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE)

    def train_epoch(self):
        """1エポック分の訓練"""
        self.model.train()
        running_loss = 0.0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def validate(self):
        """バリデーション"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return val_loss / len(self.val_loader), correct / total

    def run_training(self):
        """訓練の実行"""
        self.prepare_data()
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # 学習率スケジューラ更新
            self.scheduler.step(val_acc)
            
            # モデルの保存
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOP_PATIENCE:
                    print("Early stopping")
                    break
            
            # 結果の記録と表示
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training()