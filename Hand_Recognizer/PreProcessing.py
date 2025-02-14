import cv2
import numpy as np
import mediapipe as mp
import os
import absl.logging

# Mediapipeのログメッセージを抑制
absl.logging.set_verbosity(absl.logging.ERROR)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def preprocess_image(image):
    # 手の領域を切り出す（背景除去）
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image.shape
        x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
        x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
        y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
        y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

        hand_region = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        if hand_region.size == 0:
            return None

    # 照明補正やノイズ除去
    hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    hand_region = cv2.equalizeHist(hand_region)
    hand_region = cv2.GaussianBlur(hand_region, (5, 5), 0)

    return hand_region

def detect_edges(image):
    # Cannyエッジ検出を用いて手の輪郭や主要な線模様を抽出する
    edges = cv2.Canny(image, 100, 200)
    return edges

def extract_features(image):
    # Gaborフィルタを適用して、手の模様やテクスチャの特徴を強調する
    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)

    filtered_images = []
    for kernel in gabor_kernels:
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered_image)

    return filtered_images

def save_preprocessed_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        hand_region = preprocess_image(image)
        if hand_region is not None:
            edges = detect_edges(hand_region)
            features = extract_features(hand_region)

            # 保存
            cv2.imwrite(os.path.join(output_dir, f'hand_region_{i}.png'), hand_region)
            cv2.imwrite(os.path.join(output_dir, f'edges_{i}.png'), edges)
            for j, feature in enumerate(features):
                cv2.imwrite(os.path.join(output_dir, f'feature_{i}_{j}.png'), feature)
        else:
            print(f"手の領域が検出されませんでした: {image_path}")

# 例として入力フォルダ内のすべての画像を読み込み、前処理、エッジ検出、特徴抽出を行い、保存する
if __name__ == "__main__":
    input_dir = 'output_images_onishi/'  # ここに入力フォルダのパスを追加
    output_dir = 'onishi_preprocessed'  # ここに出力フォルダのパスを追加
    save_preprocessed_images(input_dir, output_dir)
