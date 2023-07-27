import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy


class TextClassifierModel(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float,
        num_classes: int,
        ):
        super().__init__()
        self.net = net
        self.loss_fn = nn.CrossEntropyLoss()        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters(ignore=['net'])
    
    def forward(self, batch):
        ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        return self.net(ids, attention_mask, token_type_ids)


    def feature(self, batch):
        ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        return self.net.feature(ids, attention_mask, token_type_ids)
    
    def _forward(self, batch):
        ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['label']
        predictions = self.net(ids, attention_mask, token_type_ids)
        return predictions, labels

    def training_step(self, batch, batch_idx):
        predictions, labels = self._forward(batch)
        loss = self.loss_fn(predictions, labels)
        
        self.train_accuracy.update(predictions, labels)

        self.log('train/loss', loss, prog_bar=False)
        self.log('train/acc', self.train_accuracy, prog_bar=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        predictions, labels = self._forward(batch)
        loss = self.loss_fn(predictions, labels)
        self.val_accuracy.update(predictions, labels)

        self.log('val/loss', loss.item(), prog_bar=True)
        self.log('val/acc', self.val_accuracy, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        predictions, labels = self._forward(batch)
        loss = self.loss_fn(predictions, labels)
        self.test_accuracy.update(predictions, labels)

        self.log('test/loss', loss.item(), prog_bar=True)
        self.log('test/acc', self.test_accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr = self.hparams.lr)
        return optimizer