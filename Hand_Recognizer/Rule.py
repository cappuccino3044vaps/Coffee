import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webカメラを開く
cap = cv2.VideoCapture(0)


def count_fingers(hand_landmarks):
    if not hand_landmarks: return 0

    # 各指の先端（指先: TIP）と根元（DIP, PIP）の座標
    finger_tips = [4, 8, 12, 16, 20]  # 親指, 人差し指, 中指, 薬指, 小指の先端
    finger_dips = [3, 6, 10, 14, 18]  # 各指のDIP関節
    finger_mcps = [2, 5, 9, 13, 17]   # 各指のMCP関節

    fingers = []

    for tip, dip, mcp in zip(finger_tips[1:], finger_dips[1:], finger_mcps[1:]):  # 親指以外の指
        # 3次元ベクトルを取得
        tip_pos = np.array([hand_landmarks.landmark[tip].x, 
                           hand_landmarks.landmark[tip].y,
                           hand_landmarks.landmark[tip].z])
        dip_pos = np.array([hand_landmarks.landmark[dip].x,
                           hand_landmarks.landmark[dip].y,
                           hand_landmarks.landmark[dip].z])
        mcp_pos = np.array([hand_landmarks.landmark[mcp].x,
                           hand_landmarks.landmark[mcp].y,
                           hand_landmarks.landmark[mcp].z])

        # ベクトルを計算
        vec1 = dip_pos - mcp_pos
        vec2 = tip_pos - dip_pos

        # 角度を計算
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

        if angle < 30:  # 修正: 角度が小さい時（指が伸びている）を1にする
            fingers.append(1)
        else:
            fingers.append(0)

    # 親指の判定コード（インデントを修正）
    thumb_tip = np.array([hand_landmarks.landmark[4].x, 
                        hand_landmarks.landmark[4].y,
                        hand_landmarks.landmark[4].z])
    thumb_ip = np.array([hand_landmarks.landmark[2].x,
                       hand_landmarks.landmark[2].y,
                       hand_landmarks.landmark[2].z])
    thumb_mcp = np.array([hand_landmarks.landmark[1].x,
                        hand_landmarks.landmark[1].y,
                        hand_landmarks.landmark[1].z])

    vec1 = thumb_ip - thumb_mcp
    vec2 = thumb_tip - thumb_ip
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    if angle < 30:
        fingers.insert(0, 1)
    else:
        fingers.insert(0, 0)

    return sum(fingers)  # return文のインデントを修正（ループ外に移動）

def recognize_gesture(hand_landmarks):
    """ 指の曲げ具合から簡単なジェスチャーを認識する """
    if not hand_landmarks:
        return "None"
    
    # 各指の先端（指先: TIP）と根元（DIP, PIP）の座標
    finger_tips = [4, 8, 12, 16, 20]  # 親指, 人差し指, 中指, 薬指, 小指の先端
    finger_dips = [3, 6, 10, 14, 18]  # 各指のDIP関節

    fingers = []
    
    for tip, dip in zip(finger_tips[1:], finger_dips[1:]):  # 親指以外の指
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:  # 指が伸びているかどうか
            fingers.append(1)  # 伸びている
        else:
            fingers.append(0)  # 曲がっている

    thumb_tip = hand_landmarks.landmark[finger_tips[0]]
    thumb_ip = hand_landmarks.landmark[2]
    
    # 親指の方向（右手/左手を考慮）
    if thumb_tip.x < thumb_ip.x:
        fingers.insert(0, 1)  # 親指が開いている
    else:
        fingers.insert(0, 0)  # 親指が曲がっている

    # ジェスチャー認識
    if fingers == [0, 0, 0, 0, 0]:
        return "グー（✊）"
    elif fingers == [1, 1, 1, 1, 1]:
        return "パー（🖐️）"
    elif fingers == [0, 1, 1, 0, 0]:
        return "ピース（✌️）"
    else:
        return "不明"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 画像をRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 手のランドマークを描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 指の本数を数える
            finger_count = count_fingers(hand_landmarks)
    
    # 2本指が検出された場合の処理
    if finger_count == 2:
        if two_fingers_start_time is None:
            two_fingers_start_time = time.time()
        elif time.time() - two_fingers_start_time >= 3:
            break
    else:
        two_fingers_start_time = None

    # 結果を画面に表示
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Hands Finger Count', frame)

    # qキーまたはESCキーで終了
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
