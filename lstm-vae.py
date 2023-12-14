import torch
import torch.nn as nn
import torch.optim as optim

class LSTMVAE(nn.Module):
    def __init__(self):
        super(LSTMVAE, self).__init__()
        # LSTM parameters
        self.lstm_size = 128
        self.embedding_size = 64
        self.num_layers = 1

        # Frame size
        self.frame_size = 32 * 32  # 各フレームをフラット化したサイズ

        # Encoder LSTM
        self.lstm = nn.LSTM(input_size=self.frame_size, hidden_size=self.lstm_size, num_layers=self.num_layers, batch_first=True)

        # VAE components
        self.fc_mu = nn.Linear(self.lstm_size, self.embedding_size)
        self.fc_logvar = nn.Linear(self.lstm_size, self.embedding_size)
        self.fc_decode = nn.Linear(self.embedding_size, self.lstm_size)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.frame_size, num_layers=self.num_layers, batch_first=True)

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        mu = self.fc_mu(h_n.squeeze())
        logvar = self.fc_logvar(h_n.squeeze())
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z).unsqueeze(1).repeat(1, 30, 1)  # n_frames (30) に合わせる
        output, _ = self.decoder_lstm(z)
        return output

    def forward(self, x):
        # xの形は (batch_size, n_frames, frame_size, frame_size)
        # LSTMに入力するためにフレームをフラット化する
        batch_size, n_frames, _, _ = x.size()
        x = x.view(batch_size, n_frames, -1)  # xの形を (batch_size, n_frames, frame_size*frame_size) に変更

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Model, optimizer
model = LSTMVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.view(data.size(0), data.size(1), -1)  # データのフラット化
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    average_loss = train_loss / len(train_loader.dataset)
    return average_loss


# Test loop
def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.view(data.size(0), data.size(1), -1)  # データのフラット化
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()

    average_loss = test_loss / len(test_loader.dataset)
    return average_loss

# テストデータセットの準備 (仮定)
# test_dataset = CustomDataset(test_data)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# モデルとテストデータローダーを使用してテスト
# test_loss = test(model, test_loader)
# print(f'Test Loss: {test_loss}')


def detect_anomalies(model, data_loader, threshold):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for data in data_loader:
            data = data.view(data.size(0), data.size(1), -1)  # データのフラット化
            recon_batch, _, _ = model(data)
            loss = torch.mean((recon_batch - data) ** 2, dim=[1, 2])  # 再構築誤差を計算

            # 異常検出
            anomalies.extend(loss > threshold)

    return anomalies

# 異常検知を実行
# anomaly_loader = DataLoader(anomaly_dataset, batch_size=16, shuffle=False)
# anomalies = detect_anomalies(model, anomaly_loader, threshold)

# 結果の確認
# for i, is_anomaly in enumerate(anomalies):
#     if is_anomaly:
#         print(f"Data sample {i} is an anomaly.")
