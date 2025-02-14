import torch
print(torch.__version__)  # インストールされたPyTorchのバージョンを確認
print(torch.cuda.is_available())  # CUDAが有効か確認
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
