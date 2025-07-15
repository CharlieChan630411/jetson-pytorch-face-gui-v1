# utils/face_encoder.py
# 匯入所需的套件
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# 定義 MobileFaceNet 類別，繼承自 nn.Module（PyTorch 的模型基底類別）
class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()  # 初始化父類別
        self.embedding_size = embedding_size  # 設定嵌入向量的維度（預設為 512）
        # 使用 torch.hub 從 GitHub 載入預設 MobileFaceNet 架構
        self.model = torch.hub.load(
            'Xiaoccer/MobileFaceNet_Pytorch',
            'mobilefacenet',
            pretrained=False  # 我們會手動載入訓練好的權重
        )

    def forward(self, x):
        # 前向傳遞過程：輸入張量 x，經過模型並進行 L2 正規化，回傳嵌入向量
        return F.normalize(self.model(x), p=2, dim=1)


# 載入訓練好的模型權重
# model_path: 欲載入的 .pth 檔案路徑
# device: 使用 'cuda' 或 'cpu'
def load_model(model_path, device='cuda'):
    model = MobileFaceNet()  # 建立模型物件
    # 載入權重，map_location 讓你可以在不同裝置上載入模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 設定為推論模式
    model.to(device)  # 將模型移至指定的裝置（GPU 或 CPU）
    return model


# 處理單張臉部圖像，將其轉換為模型可接受的格式
# image: BGR 格式的圖像（OpenCV 預設格式）
def preprocess_face(image):
    face = cv2.resize(image, (112, 112))  # 調整為 112x112（MobileFaceNet 輸入尺寸）
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 轉為 RGB（PyTorch 預期格式）
    face = face / 255.0  # 將像素值縮放至 [0, 1]
    face = (face - 0.5) / 0.5  # 標準化為 [-1, 1]
    face = torch.from_numpy(face.transpose(2, 0, 1)).float()  # 將 HWC 轉為 CHW，並轉為 float tensor
    face = face.unsqueeze(0)  # 增加 batch 維度（[C,H,W] → [1,C,H,W]）
    return face


# 執行臉部編碼，將圖像轉為一個 512 維向量
# model: 已載入權重的模型
# image: 一張 BGR 圖像（臉部特寫）
# device: 'cuda' 或 'cpu'
def get_face_embedding(model, image, device='cuda'):
    face_tensor = preprocess_face(image).to(device)  # 預處理並轉至指定裝置
    with torch.no_grad():  # 停用梯度以節省記憶體（推論模式）
        embedding = model(face_tensor)  # 執行前向傳遞
        embedding = embedding.cpu().numpy().flatten()  # 將向量轉為一維 NumPy 陣列
    return embedding  # 回傳臉部特徵向量

