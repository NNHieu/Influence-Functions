import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from tqdm import tqdm, trange
import yaml
import os
import numpy as np
import copy
from pytorch_lightning.callbacks import BaseFinetuning

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.net.bert)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        pass

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits.detach(), dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss.mean()

def train_bias_model(config, model, loader, test_loader, device, fp16_run=False):
    n_epochs = config['n_epochs']
    optim = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    model.to(device)
    best_acc = 0
    loss_fn = GeneralizedCELoss()
    scaler = GradScaler(enabled=fp16_run)
    for epoch in range(n_epochs):
        model.train()
        for img, labels in (pbar := tqdm(loader)):
            img = img.to(device)
            labels = labels.to(device)
            with autocast(enabled=fp16_run):
                output = model(img)
            loss = loss_fn(output, labels).mean()
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            # optim.step()
            scaler.step(optim)
            scaler.update()
            pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}')
        print(f'Accuracy: {acc}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'log/pgd/{config["model_name"]}_bias_{config["dataset"]}.pth')
            best_model = copy.deepcopy(model)
    return best_model

class OversamplingDataset(Dataset):

    def __init__(self, dataset, weight):
        self.dataset = dataset
        self.weight = weight
        assert len(weight) == len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sampled_idx = torch.multinomial(self.weight, 1)
        return self.dataset[sampled_idx]

def get_bias_weight(model, dataset, device):
    model.eval()
    model.to(device)
    weight = []
    for i in trange(len(dataset)):
        data, label = dataset[i]
        data = data.unsqueeze(0).to(device)
        label = torch.tensor([label], device=device)
        model.fc.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, label)
        loss.backward()
        grad_norm = sum([param.grad.norm() for param in model.fc.parameters()])
        weight.append(grad_norm)
    return torch.stack(weight)
