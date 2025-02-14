import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def preprocess_image(image, size=(128, 128)):
    image = cv2.resize(image, size)
    image = image / 255.0  # 正規化
    return image

def load_dataset(dataset_path, img_size=(224, 224)):
    X, y = [], []
    
    # データ拡張用の変換
    augment = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    for label, category in enumerate(["negative", "positive"]):
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            image = cv2.imread(file_path)
            if image is not None:
                # リサイズと正規化
                image = cv2.resize(image, img_size)
                image = image / 255.0
                
                # データ拡張
                for _ in range(5):  # 各画像から5つのバリエーションを作成
                    augmented = augment(image)
                    X.append(augmented)
                    y.append(label)
    
    return np.array(X), np.array(y)
