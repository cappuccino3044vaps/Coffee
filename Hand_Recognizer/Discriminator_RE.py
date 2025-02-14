import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import mediapipe as mp
from Train_CNN import PalmRecognizer, extract_palm_region

class DiscriminatorConfig:
    """識別器の設定値を管理するクラス"""
    # モデル関連
    MODEL_PATH = 'best_model.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 画像処理関連
    IMG_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # 推論関連
    CONFIDENCE_THRESHOLD = 0.73
    MIN_HAND_SIZE = 50  # 最小手のひらサイズ(ピクセル)

class ImageProcessor:
    """画像処理関連のユーティリティクラス"""
    @staticmethod
    def preprocess_frame(frame):
        """フレームの前処理"""
        hand_region = extract_palm_region(frame)
        if hand_region is None:
            return None

        pil_image = Image.fromarray(hand_region)
        preprocess = transforms.Compose([
            transforms.Resize((DiscriminatorConfig.IMG_SIZE, DiscriminatorConfig.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                DiscriminatorConfig.NORMALIZE_MEAN,
                DiscriminatorConfig.NORMALIZE_STD
            )
        ])
        return preprocess(pil_image).unsqueeze(0)

    @staticmethod
    def extract_hand_region(frame):
        """手のひら領域を抽出"""
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_hand_landmarks[0].landmark
                x_coords = [int(lm.x * w) for lm in landmarks]
                y_coords = [int(lm.y * h) for lm in landmarks]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 最小サイズチェック
                if (x_max - x_min) < DiscriminatorConfig.MIN_HAND_SIZE or \
                   (y_max - y_min) < DiscriminatorConfig.MIN_HAND_SIZE:
                    return None, None
                    
                return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
        return None, None

class Discriminator:
    """手のひら識別器クラス"""
    def __init__(self):
        self.device = torch.device(DiscriminatorConfig.DEVICE)
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """モデルをロード"""
        model = PalmRecognizer()
        model.load_state_dict(torch.load(
            DiscriminatorConfig.MODEL_PATH,
            map_location=self.device, weights_only=True
        ))
        return model.to(self.device)

    def predict(self, frame):
        """フレームに対して推論を実行"""
        input_tensor = ImageProcessor.preprocess_frame(frame)
        if input_tensor is None:
            return "No hand detected", 0.0
        
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            prob = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(prob, dim=1)
            prediction_label = "Positive" if prediction.item() == 1 and \
                confidence.item() > DiscriminatorConfig.CONFIDENCE_THRESHOLD else "Negative"
            return prediction_label, confidence.item()

class VideoProcessor:
    """ビデオ処理クラス"""
    def __init__(self):
        self.discriminator = Discriminator()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("カメラが開けませんでした。")

    def process_frame(self, frame):
        """フレームを処理"""
        hand_region, bbox = ImageProcessor.extract_hand_region(frame)
        if hand_region is not None and bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            prediction, confidence = self.discriminator.predict(hand_region)
            
            # 結果を表示
            cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return frame

    def run(self):
        """メインループ"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("フレーム取得に失敗しました。")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('WebCam Input', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        processor = VideoProcessor()
        processor.run()
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")