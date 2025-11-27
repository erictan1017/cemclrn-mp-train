import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from modules.models.SigNetCNN import SigNetCNN

MARGIN = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
FUZZY = 1e-8
GAMMA = 0.1
SCHEDULER_MILESTONES = [5, 10]  # 70% and 90% of total epochs (20)


# See https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/loss.py
# Based on Eq. 1 of SigNet paper (https://arxiv.org/pdf/1707.02131)
# Values for α and β are not mentioned, so we use 1
def contrastive_loss(output1, output2, y):
    euclidean_distance = F.pairwise_distance(output1, output2)
    contrastive_loss = (1 - y) * euclidean_distance**2 + y * (
        torch.max(torch.zeros_like(euclidean_distance), MARGIN - euclidean_distance)
        ** 2
    )
    contrastive_loss = torch.mean(contrastive_loss, dtype=torch.float)

    return contrastive_loss, euclidean_distance


class SigNetSiamese(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, margin=1.0):
        super().__init__()

        # "Branch" CNN of the Siamese network
        self.cnn = SigNetCNN()

    def forward(self, x1, x2):
        y1 = self.cnn(x1)
        y2 = self.cnn(x2)
        return y1, y2

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        output1, output2 = self(x1, x2)
        loss = contrastive_loss(output1, output2, y)[0]

        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        output1, output2 = self(x1, x2)
        loss = contrastive_loss(output1, output2, y)[0]

        # Log validation loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    # We skip defining validation_step for now, as we train with a fixed number of epochs instead.

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.cnn.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM,
            eps=FUZZY,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

        return optimizer
