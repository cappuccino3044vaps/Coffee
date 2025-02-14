import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# MediaPipeの初期設定（RuleNML.pyと同一）
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 特徴量抽出関数（RuleNML.pyと完全一致）
def extract_features(hand_landmarks):
    """ 指の関節の角度と長さを特徴量として抽出 """
    features = []
    finger_indices = [
        (1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)
    ]
    
    for base, mid, tip in finger_indices:
        base_3d = np.array([hand_landmarks.landmark[base].x, hand_landmarks.landmark[base].y, hand_landmarks.landmark[base].z])
        mid_3d = np.array([hand_landmarks.landmark[mid].x, hand_landmarks.landmark[mid].y, hand_landmarks.landmark[mid].z])
        tip_3d = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])

        vec1 = mid_3d - base_3d
        vec2 = tip_3d - mid_3d
        
        length1 = np.linalg.norm(vec1)
        length2 = np.linalg.norm(vec2)
        angle = np.arccos(np.dot(vec1, vec2) / (length1 * length2 + 1e-6))
        
        dir1 = vec1 / (length1 + 1e-6)
        dir2 = vec2 / (length2 + 1e-6)
        
        features.extend([length1, length2, angle, dir1[0], dir1[1], dir1[2], dir2[0], dir2[1], dir2[2]])
    
    return features

# データ収集関数
def collect_training_data(gesture_classes=5, samples_per_class=30):
    """Webカメラからトレーニングデータを収集"""
    cap = cv2.VideoCapture(0)
    data = []
    labels = []
    
    try:
        for class_id in range(gesture_classes):
            print(f"Collecting data for gesture {class_id}...")
            input(f"Gesture {class_id} の準備ができたらEnterを押してください...")
            
            for _ in range(samples_per_class):
                ret, frame = cap.read()
                if not ret: continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features = extract_features(hand_landmarks)
                    data.append(features)
                    labels.append(class_id)
                    
                cv2.putText(frame, f"Collecting: Gesture {class_id}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(100)
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return np.array(data), np.array(labels)

# メイン処理
if __name__ == "__main__":
    # データ収集（既存データがあれば読み込み）
    if os.path.exists('gesture_dataset.csv'):
        df = pd.read_csv('gesture_dataset.csv')
        X = df.drop('label', axis=1).values
        y = df['label'].values
    else:
        X, y = collect_training_data()
        pd.DataFrame(np.column_stack((X, y))).to_csv('gesture_dataset.csv', index=False)
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoostモデルの設定（RuleNML.pyと互換）
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        objective='multi:softmax',
        num_class=5
    )
    
    # モデル訓練
    model.fit(X_train, y_train)
    
    # 精度検証
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    # モデル保存（RuleNML.pyで使用するファイル名）
    model.save_model("gesture_model.json")
    print("Model saved as gesture_model.json")