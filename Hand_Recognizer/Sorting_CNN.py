import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# モデルの定義
def create_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 0 -> 手のひら反対, 1 -> 手のひらカメラ向き
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# データセットの準備
def load_images_from_folder(folder, img_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            # ラベル設定: 例えば手のひらがカメラ向きなら1、反対なら0
            label = 1 if 'palm_facing' in filename else 0
            labels.append(label)
    return np.array(images), np.array(labels)

# データのロード
train_images, train_labels = load_images_from_folder('path_to_train_images')
test_images, test_labels = load_images_from_folder('path_to_test_images')

# モデル作成と訓練
model = create_model()
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# モデルの保存
model.save('hand_palm_orientation_model.h5')
