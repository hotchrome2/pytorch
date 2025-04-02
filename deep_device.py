import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gstools import SRF, Gaussian
from gstools.random import MasterRNG
from scipy import integrate

m = 100
n = 110
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データ生成
x = np.linspace(0, 1, m)
seed = MasterRNG(20170519)
def one_function(x):
    model = Gaussian(dim=1, var=3, len_scale=0.3)
    srf = SRF(model, seed=seed())
    f = srf.structured([x])
    return f

U = np.zeros((m, n))
for i in range(m):
    U[:, i] = one_function(x)

def integrate_one_function(f):
    result = np.zeros_like(f)
    result[0] = 0
    for i in range(1, len(f)):
        result[i] = integrate.simpson(f[:i+1], x=x[:i+1])
    return result

S = np.zeros((m, n))
for i in range(n):
    S[:, i] = integrate_one_function(U[:, i])

# カスタムDatasetの定義
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, U, x, S):
        self.U = torch.tensor(U, dtype=torch.float32).T
        self.x = torch.tensor(np.tile(x, n).reshape(m * n, 1), dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32).T.reshape(m * n, 1)

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.U[idx], self.x[idx], self.S[idx]

# データセットとデータローダーの作成
dataset = CustomDataset(U, x, S)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# モデル定義
class DeepONet(nn.Module):
    def __init__(self, neurons=40, in1=m, in2=1, output_neurons=20):
        super(DeepONet, self).__init__()
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.b1 = nn.Sequential(
            nn.Linear(in1, neurons), nn.ReLU(),
            nn.Linear(neurons, neurons), nn.ReLU(),
            nn.Linear(neurons, output_neurons)
        )
        self.b2 = nn.Sequential(
            nn.Linear(in2, neurons), nn.ReLU(),
            nn.Linear(neurons, neurons), nn.ReLU(),
            nn.Linear(neurons, neurons), nn.ReLU(),
            nn.Linear(neurons, output_neurons)
        )

    def forward(self, x1, x2):
        x1 = self.b1(x1)
        x2 = self.b2(x2)
        x = torch.sum(x1 * x2, dim=1).unsqueeze(1) + self.b # torch.einsumの代わりにtorch.sumを使用
        return x

model = DeepONet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 学習
train_losses = []
epochs = 2000
for i in range(epochs):
    model.train()
    l_total = 0
    for u_, x_, s_ in train_loader:
        u_, x_, s_ = u_.to(device), x_.to(device), s_.to(device)
        optimizer.zero_grad()
        y_pred = model(u_, x_)
        l = torch.mean((y_pred - s_)**2)
        l_total += l.item()
        l.backward()
        optimizer.step()
    l_total /= len(train_loader)
    train_losses.append(l_total)
    scheduler.step(l_total)
    if i % 20 == 0:
        print(f'Epoch: {i:>4d}/{epochs}, Loss: {l_total:6f}')

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.show()

# 検証
model.eval()
x_t = torch.tensor(x, dtype=torch.float32).to(device)
u_t = torch.cos(5 * x_t).cpu().numpy()
s_t = 1/5 * torch.sin(5 * x_t).cpu().numpy()
u_t_ = torch.tensor(u_t, dtype=torch.float32).unsqueeze(0).to(device)
p = model(u_t_, x_t).cpu().detach().numpy()

plt.plot(x, u_t[0], label="Func")
plt.plot(x, s_t, label="integration analytical")
plt.plot(x, p[0], label="integration model")
plt.legend()
plt.show()
