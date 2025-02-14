import torch
import time
from Train_CNN import PalmRecognizer  # モデル定義をインポート

# デバイス確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデル定義
model = PalmRecognizer(num_classes=2)  # クラス数を指定

# 時間計測開始
start_time = time.time()

# モデルのパラメータをロード
model.load_state_dict(torch.load("palm_recognizer.pth", map_location=device))

# GPUに転送
model.to(device)

# 時間計測終了
end_time = time.time()

print(f"Model loaded in {end_time - start_time:.4f} seconds")
