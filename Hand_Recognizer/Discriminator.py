import cv2
import torch
from torchvision import transforms
from PIL import Image
from Train_CNN import PalmRecognizer
from Train_CNN import extract_palm_region
import numpy as np
import mediapipe as mp

def preprocess_frame(frame):
    # 手のひら領域検出
    hand_region = extract_palm_region(frame)
    if hand_region is None:
        return None

    # PIL 画像に変換
    pil_image = Image.fromarray(hand_region)

    # 前処理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return preprocess(pil_image).unsqueeze(0)  # バッチ次元を追加

def predict(frame, model, device):
    # 前処理
    input_tensor = preprocess_frame(frame)
    if input_tensor is None:
        return "No hand detected", 0.0
    
    # 推論
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        confidence = prob[0][1].item()
        return "Positive" if prob[0][1] > 0.95 else "Negative", confidence

def extract_hand_region(frame):
    # BGRからRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 手の検出
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # 手のランドマークからバウンディングボックスを計算
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        # バウンディングボックスの計算
        x_min = int(min(x_coords) * frame.shape[1])
        x_max = int(max(x_coords) * frame.shape[1])
        y_min = int(min(y_coords) * frame.shape[0])
        y_max = int(max(y_coords) * frame.shape[0])
        
        # 余白を追加
        margin = 20
        x_min = max(0, x_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(frame.shape[0], y_max + margin)
                # 手のひら領域のサイズチェックを追加
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return None, None
        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    return None, None

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# モデル読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PalmRecognizer(num_classes=2).to(device)  # 2クラス分類に変更（手のひらかそうでないか）
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Webカメラの設定
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けませんでした。")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得に失敗しました。")
        break

    # 手のひら領域を検出
    hand_region, bbox = extract_hand_region(frame)
    if hand_region is not None and bbox is not None:
        x_min, y_min, x_max, y_max = bbox  # バウンディングボックス座標を取得
        # 推論
        prediction, confidence = predict(hand_region, model, device)
        
        # 結果を映像上に表示
        cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 手のひら領域を表示
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    cv2.imshow('WebCam Input', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
