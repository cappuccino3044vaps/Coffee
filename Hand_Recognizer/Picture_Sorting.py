import cv2
import mediapipe as mp
import numpy as np
import os

# 入出力フォルダ
input_folder = "outputimages8"
output_folder = "dataset2/positive/"
os.makedirs(output_folder, exist_ok=True)

# MediaPipe Hands のセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8)

def calculate_hand_rotation(landmarks):
    """手の回転を計算する"""
    # 手首を原点とする
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x,
                      landmarks[mp_hands.HandLandmark.WRIST].y,
                      landmarks[mp_hands.HandLandmark.WRIST].z])
    
    # 主要なランドマークの位置を取得
    middle_mcp = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                          landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                          landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z]) - wrist
    
    index_mcp = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].z]) - wrist
    
    pinky_mcp = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_MCP].y,
                         landmarks[mp_hands.HandLandmark.PINKY_MCP].z]) - wrist
    
    # 手のひらの平面を定義する基底ベクトルを計算
    v1 = middle_mcp  # 手首から中指MCPへのベクトル
    v2 = pinky_mcp - index_mcp  # 人差し指MCPから小指MCPへのベクトル
    
    # 手のひらの法線ベクトルを計算
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    return normal

def is_palm_facing_camera(landmarks):
    """手のひらがカメラを向いているかを判定する（回転を考慮）"""
    # 手の回転を考慮した法線ベクトルを取得
    normal = calculate_hand_rotation(landmarks)
    
    # カメラの視線ベクトル（カメラから手のひらに向かうベクトル）
    camera_direction = np.array([0, 0, -1])
    
    # 法線ベクトルとカメラの視線ベクトルの内積を計算
    dot_product = np.dot(normal, camera_direction)
    
    # 閾値を設定して2値分類（手のひらが見えるか見えないか）
    threshold = 0.2  # この閾値は調整可能
    return dot_product > threshold

def landmarks_within_image(landmarks, image_shape):
    """ランドマークが画像の範囲内に収まっているかを確認する"""
    height, width, _ = image_shape
    for landmark in landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
    return True

def count_landmarks_in_image(image, hand_landmarks):
    """アノテーション画像中のランドマークの点の数を数える"""
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    num_points = cv2.countNonZero(binary)
    return num_points

def draw_hand_annotations(image, hand_landmarks, is_palm_facing):
    """手のランドマークとアノテーションを描画する"""
    # 全ての関節点が検出されているか確認
    if not all_landmarks_detected(hand_landmarks.landmark):
        #print("All landmarks not detected")
        return None  # 全ての関節点が検出されていない場合はNoneを返す

    # ランドマークが画像の範囲内に収まっているか確認
    if not landmarks_within_image(hand_landmarks.landmark, image.shape):
        #print("Landmarks not within image")
        return None  # ランドマークが画像の範囲外にある場合はNoneを返す

    # ランドマークの描画
    '''
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    '''
    # 手のひらの向きの状態を表示
    status = "Palm Facing Camera" if is_palm_facing else "Palm Away"
    color = (0, 255, 0) if is_palm_facing else (0, 0, 255)  # 緑か赤
    
    # テキストを描画（画像の上部に配置）
    #cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return image

def all_landmarks_detected(landmarks):
    """全ての関節点が検出されたかを確認する"""
    for landmark in landmarks:
        if landmark.x == 0 and landmark.y == 0 and landmark.z == 0:  # 座標が全て0の場合は検出されていないとみなす
            return False
    return True

def is_blurry(image, threshold=30.0):
    """画像がボケているかを判定する"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def get_unique_filename(output_folder, filename):
    """出力フォルダに同じ名前のファイルがある場合、ユニークな名前を生成する"""
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(output_folder, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    return unique_filename

def process_images():
    """画像を処理して手のひらの向きを分類し、アノテーション付きで保存する"""
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        if is_blurry(image):
            continue  # 画像がボケている場合はスキップ

        # BGR -> RGB 変換（MediaPipe用）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 手の検出
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #print(f"Hand landmarks detected in image: {filename}")
                is_palm_facing = is_palm_facing_camera(hand_landmarks.landmark)
                
                # アノテーションを描画
                annotated_image = draw_hand_annotations(image, hand_landmarks, is_palm_facing)
                
                if annotated_image is not None and is_palm_facing:
                    num_points = count_landmarks_in_image(image, hand_landmarks)
                    if num_points < 21:  # 21は手のランドマークの数
                        #print(f"Not all landmarks are visible in image: {filename}")
                        continue  # 全てのランドマークが見えていない場合はスキップ

                    unique_filename = get_unique_filename(output_folder, filename)
                    output_path = os.path.join(output_folder, unique_filename)
                    cv2.imwrite(output_path, annotated_image)
                    break  # 1つの手が手のひら向きなら保存

if __name__ == "__main__":
    process_images()
    print("処理完了: 画像を出力しました。")