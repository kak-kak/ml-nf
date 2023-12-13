import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transforms:
            sample = self.transforms(sample)

        return sample
      
# 例として、Numpy配列を作成（ここではランダムなデータを使用）
np_data = np.random.randn(100, 32, 32)  # 100個の32x32のデータ

# PyTorchのテンソルに変換するための変換関数
transforms = torch.from_numpy

# カスタムデータセットのインスタンスを作成
custom_dataset = CustomDataset(np_data, transforms=transforms)

# DataLoaderを作成
dataloader = DataLoader(custom_dataset, batch_size=10, shuffle=True)

# DataLoaderを使用してデータをイテレート
for batch in dataloader:
    # ここで、バッチ処理されたデータに対する操作を行います
    pass


import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# VAEモデルの定義
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 20) # 10 for mean and 10 for log variance
        )

        # デコーダ
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 32*32),
            nn.Sigmoid() # ピクセル値は0と1の間
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # エンコーダからの出力
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)

        # reparameterization trick
        z = self.reparameterize(mu, logvar)

        # デコーダからの再構成
        return self.decoder(z), mu, logvar

# 損失関数
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# モデル、最適化手法の設定
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# データの前処理とロード
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 訓練ループ
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {train_loss / len(dataloader.dataset)}')

# 訓練の実行
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch)


# 異常検知の閾値
anomaly_threshold = 0.005

# テストデータセットの準備
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def detect_anomalies(model, test_loader, threshold):
    model.eval()  # モデルを評価モードに設定
    anomalies = []

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.view(data.size(0), -1)
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)

            # バッチ内の各サンプルについて異常かどうかを判断
            for j in range(data.size(0)):
                sample_loss = loss[j].item()
                if sample_loss > threshold:
                    anomalies.append((i*test_loader.batch_size) + j)

    return anomalies

# 異常検知の実行
anomalies = detect_anomalies(model, test_loader, anomaly_threshold)

print(f'検出された異常サンプルの数: {len(anomalies)}')
print('異常と判定されたサンプルのインデックス:', anomalies)

