import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb

# MediaPipeのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# XGBoostモデルのロード
model = xgb.XGBClassifier()
booster = xgb.Booster()
booster.load_model("gesture_model.json")
model._Booster = booster

# 特徴量の抽出（前述の関数を再利用）
def extract_features(hand_landmarks):
    """ 指の関節の角度と長さを特徴量として抽出 """
    features = []
    # 各指の関節ポイント（親指、人差し指、中指、薬指、小指）
    finger_indices = [
        (1, 2, 4),    # 親指（MCP, IP, TIP）
        (5, 6, 8),    # 人差し指（MCP, DIP, TIP）
        (9, 10, 12),  # 中指
        (13, 14, 16), # 薬指
        (17, 18, 20)  # 小指
    ]
    
    for base, mid, tip in finger_indices:
        # 3次元座標を取得
        base_3d = np.array([hand_landmarks.landmark[base].x, 
                           hand_landmarks.landmark[base].y,
                           hand_landmarks.landmark[base].z])
        mid_3d = np.array([hand_landmarks.landmark[mid].x,
                          hand_landmarks.landmark[mid].y,
                          hand_landmarks.landmark[mid].z])
        tip_3d = np.array([hand_landmarks.landmark[tip].x,
                          hand_landmarks.landmark[tip].y,
                          hand_landmarks.landmark[tip].z])

        # ベクトル計算（関節間）
        vec1 = mid_3d - base_3d
        vec2 = tip_3d - mid_3d
        
        # 特徴量計算
        length1 = np.linalg.norm(vec1)  # 根元から中間関節までの長さ
        length2 = np.linalg.norm(vec2)  # 中間関節から先端までの長さ
        angle = np.arccos(np.dot(vec1, vec2) / (length1 * length2 + 1e-6))  # 関節角度
        
        # 追加特徴量（ベクトルの方向成分）
        dir1 = vec1 / (length1 + 1e-6)
        dir2 = vec2 / (length2 + 1e-6)
        
        features.extend([
            length1, length2, angle,
            dir1[0], dir1[1], dir1[2],  # ベクトル方向の成分
            dir2[0], dir2[1], dir2[2]
        ])
    
    return features

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = extract_features(hand_landmarks)
            prediction = model.predict([features])[0]
            cv2.putText(frame, f"Gesture: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
