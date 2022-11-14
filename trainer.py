import torch
import torch.nn as nn
import tqdm
import os

from abc import ABC, abstractmethod

from utils import GazeEarlyStopping
from utils import mask_mse_loss
from utils import LOGGER

class Trainer(ABC):
    def __init__(self, n_epochs, model, train_dl, eval_dir, early_stop, task, device):
        self.model = model
        self.train_dl = train_dl
        self.eval_dir = eval_dir
        self.early_stop = early_stop
        self.n_epochs = n_epochs
        self.task = task
        self.device = device

        # if not os.path.exists(eval_dir):
        #    os.makedirs(eval_dir)

       # self.writer = SummaryWriter(self.eval_dir)

    @abstractmethod
    def train_one_step(self, batch):
        pass

    def train(self):
        n_batches_one_epoch = len(self.train_dl)
        n_params = sum(p.numel() for p in self.model.parameters())
        # mlflow.log_metric("n_params", n_params)
        LOGGER.info(f"Num epochs: {self.n_epochs}")
        LOGGER.info(f"Num parameters: {n_params}")
        LOGGER.info(f"Begin training task {self.task}")

        self.model.to(self.device)
        self.model.train()

        epoch_loss_ls = []
        it = 1

        for epoch in tqdm(range(1, self.n_epochs + 1)):
            for batch in tqdm(self.train_dl):
                it += 1

                loss = self.train_one_step(batch)
                # self.writer.add_scalar("train/loss", loss, it)
                epoch_loss_ls.append(loss)

            epoch_loss_avg = sum(epoch_loss_ls) / len(epoch_loss_ls)
            epoch_loss_ls = []
            LOGGER.info(f"Done epoch {epoch} / {self.n_epochs}")
            LOGGER.info(f"Avg loss epoch {epoch}: {epoch_loss_avg:.4f}")

            self.early_stop()

            # for key, metric in self.early_stop.tester.metrics.items():
                # self.writer.add_scalar(f"val/{key}", metric, it // n_batches_one_epoch)

            if self.early_stop.stop:
                break


class GazeTrainer(Trainer):
    def __init__(self, n_epocs, max_grad_norm, patience, model, train_dl, val_dl, optim, scheduler, eval_dir,
                 task, device, monitor, monitor_mode):
        early_stop = GazeEarlyStopping(patience, model, val_dl, eval_dir, device, task, monitor, monitor_mode)
        super().__init__(n_epocs, model, train_dl, eval_dir, early_stop, task, device)

        self.optim = optim
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.target_pad = train_dl.target_pad

        self.criterion = nn.MSELoss(reduction="mean")

    def train_one_step(self, batch):
        self.model.zero_grad()

        b_input, b_target, b_mask = batch
        b_input = b_input.to(self.device)
        b_target = b_target.to(self.device)
        b_mask = b_mask.to(self.device)

        b_output = self.model(input_ids=b_input, attention_mask=b_mask)[0]

        active_outputs, active_targets = mask_mse_loss(b_output, b_target, self.target_pad, self.model.d_out)
        loss = self.criterion(active_outputs, active_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
        self.optim.step()
        self.scheduler.step()

        return loss.item()