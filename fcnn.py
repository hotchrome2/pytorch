import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 周波数
frequency = 1.5

num_samples = 100
times = np.linspace(0, 10, num_samples)
amplitudes = np.sin(frequency * times)  # 定数周波数に基づいて正弦波を生成

x_train = torch.tensor(times, dtype=torch.float32).unsqueeze(1)  # 時間を入力
y_train = torch.tensor(amplitudes, dtype=torch.float32).unsqueeze(1)

class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)  # 1:入力は時間のみ。周波数は入力としない
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x

model = SinModel()  # モデルのインスタンス化

# 損失関数と最適化アルゴリズムの定義
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学習
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 予測
model.eval()
with torch.no_grad():
    y_pred = model(x_train)

# プロット
plt.plot(times, amplitudes, label='Actual')
plt.plot(times, y_pred, label='Predicted')
plt.legend()
plt.show()



