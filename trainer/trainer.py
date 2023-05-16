import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch import Tensor


class AutoencoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.config = config

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def detect_anomaly(self, x: Tensor):
        rec = self(x)
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }

    def training_step(self, batch: Tensor, batch_idx):
        x, _ = batch
        recon = self(x)
        loss = nn.MSELoss()(recon, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), lr=self.config['lr'])
