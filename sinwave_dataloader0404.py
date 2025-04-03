"""
データセット分割と損失の計算
- バッチごとに損失を計算し、累積するように変更しました。
- テストデータ全体の平均損失を出力するように変更しました。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ハイパーパラメータ
input_size = 2  # 特徴量が w, x の2つ
hidden_size = 32
output_size = 1
learning_rate = 0.01
epochs = 1000
batch_size = 32

# モデル定義
class SineWavePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SineWavePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SineWavePredictor(input_size, hidden_size, output_size)

# 損失関数と最適化関数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# データ生成
def generate_data(num_samples):
    x = np.linspace(0, 2 * np.pi, num_samples)
    w = np.random.uniform(1, 5, num_samples)
    y = np.sin(w * x)
    features = np.stack((w, x), axis=-1)
    features = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return features, y

all_features, all_y = generate_data(150)

# データセットの分割 (インデックスを使用)
dataset = TensorDataset(all_features, all_y)
train_size = int(0.7 * len(dataset))
train_dataset = TensorDataset(all_features[:train_size], all_y[:train_size])
test_dataset = TensorDataset(all_features[train_size:], all_y[train_size:])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 学習
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_features, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}')

# テスト
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_features, batch_y in test_loader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')

# プロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
test_features_all = torch.cat([features for features, _ in test_loader])
test_y_all = torch.cat([labels for _, labels in test_loader])
test_outputs_all = model(test_features_all)
plt.plot(test_features_all[:, 1].numpy(), test_y_all.numpy(), label='True')
plt.plot(test_features_all[:, 1].numpy(), test_outputs_all.detach().numpy(), label='Predicted') # detach()を追加
plt.legend()
plt.title('Prediction Result')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
