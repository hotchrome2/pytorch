import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ハイパーパラメータ
input_size = 2  # 特徴量が w, x の2つ
hidden_size = 32
output_size = 1
learning_rate = 0.01
epochs = 1000

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
    w = np.random.uniform(1, 5, num_samples)  # ランダムな角周波数 w を生成
    y = np.sin(w * x)
    features = np.stack((w, x), axis=-1)  # w と x を結合
    features = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return features, y

train_features, train_y = generate_data(100)
test_features, test_y = generate_data(50)

# 学習
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# テスト
model.eval()
with torch.no_grad():
    test_outputs = model(test_features)
    test_loss = criterion(test_outputs, test_y)
    print(f'Test Loss: {test_loss.item():.4f}')

# プロット
plt.plot(test_features[:, 1].numpy(), test_y.numpy(), label='True')
plt.plot(test_features[:, 1].numpy(), test_outputs.numpy(), label='Predicted')
plt.legend()
plt.show()
