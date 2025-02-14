import cv2
import os
import mediapipe as mp

# 保存先のフォルダ
output_folder = 'outputimages8'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)  # 信頼度の閾値

# カメラをオープン
cap = cv2.VideoCapture(0)

# カメラが正常に開けたか確認
if not cap.isOpened():
    print("カメラが開けませんでした。")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("フレームの取得に失敗しました。")
        break

    # BGRからRGBに変換(MediaPipeはRGB形式を期待している)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 手のランドマークを検出
    results = hands.process(rgb_frame)

    # 手が検出された場合
    if results.multi_hand_landmarks:
        # 画像ファイルを保存
        filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        print(f"フレーム {frame_count} を保存しました。")
        frame_count += 1

    # フレームを表示
    cv2.imshow('Camera', frame)

    # 'q'キーでキャプチャを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# カメラを解放
cap.release()

# ウィンドウを閉じる
cv2.destroyAllWindows()

print(f"{frame_count} フレームを保存しました。")
