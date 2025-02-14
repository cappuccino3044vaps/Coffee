import torch


#cudaが使えるか
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)