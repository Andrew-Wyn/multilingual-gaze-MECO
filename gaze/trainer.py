import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod
from gaze.tester import GazeTester
from gaze.utils import LOGGER, mask_mse_loss
from gaze.utils import LOGGER, create_finetuning_optimizer, create_scheduler, randomize_model, minMaxScaling, load_model_from_hf
from modeling.custom_bert import BertForTokenClassification
from gaze.dataloader import GazeDataLoader

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification
)

class Trainer(ABC):
    def __init__(self, cf, model, train_dl, tester, task, device, writer):
        self.model = model
        self.train_dl = train_dl
        self.n_epochs = cf.n_epochs
        self.task = task
        self.device = device
        self.writer = writer
        self.tester = tester
        self.cf = cf

    @abstractmethod
    def train_one_step(self, batch):
        pass

    def train(self, save_model=False, output_dir=None):
        n_batches_one_epoch = len(self.train_dl)
        n_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f"Num epochs: {self.n_epochs}")
        LOGGER.info(f"Num parameters: {n_params}")
        LOGGER.info(f"Begin training task {self.task}")

        self.model.to(self.device)
        self.model.train()

        it = 1

        for _ in tqdm(range(1, self.n_epochs + 1)):
            for batch in self.train_dl:
                it += 1

                loss = self.train_one_step(batch)
                self.writer.add_scalar(f"{self.task}/train/loss_step_wise", loss, it)

            self.tester.evaluate()

            for key, metric in self.tester.train_metrics.items():
                self.writer.add_scalar(f"{self.task}/train/{key}", metric, it // n_batches_one_epoch)
            
            if not self.tester.test_dl is None: 
                for key, metric in self.tester.test_metrics.items():
                    self.writer.add_scalar(f"{self.task}/test/{key}", metric, it // n_batches_one_epoch)

        LOGGER.info(f"Training Done -> Train Loss_all : {self.tester.train_metrics['loss_all']}")
        if not self.tester.test_dl is None:
            LOGGER.info(f"Training Done -> Test Loss_all : {self.tester.test_metrics['loss_all']}")

        # save the model after last epoch
        if save_model:
            folder_name = os.path.join(output_dir, "model-"+self.cf.model_name+"-finetuned")
            
            if self.cf.random_weights:
                folder_name = folder_name + "randomized"
            else:
                folder_name = folder_name + "pretrained"

            if self.cf.full_finetuning:
                folder_name = folder_name + "-full"
            else:
                folder_name = folder_name + "-notfull"

            self.model.save_pretrained(folder_name)


class GazeTrainer(Trainer):
    def __init__(self, cf, model, train_dl, optim, scheduler,
                 task, device, writer, test_dl=None):
        tester = GazeTester(model, device, task, train_dl, test_dl)
        super().__init__(cf, model, train_dl, tester, task, device, writer)

        self.optim = optim
        self.scheduler = scheduler
        self.max_grad_norm = cf.max_grad_norm
        self.target_pad = train_dl.target_pad

        self.criterion = nn.MSELoss(reduction="mean")

    def train_one_step(self, batch):
        self.model.zero_grad()

        b_input, b_target, b_mask = batch
        b_input = b_input.to(self.device)
        b_target = b_target.to(self.device)
        b_mask = b_mask.to(self.device)

        b_output = self.model(input_ids=b_input, attention_mask=b_mask)[0]
        
        active_outputs, active_targets = mask_mse_loss(b_output, b_target, self.target_pad, self.model.num_labels)
        loss = self.criterion(active_outputs, active_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
        self.optim.step()
        self.scheduler.step()

        return loss.item()


def cross_validation(cf, d, writer, DEVICE, k_folds=10):
    """
    Perform a k-fold cross-validation
    """

    l = len(d.text_inputs)
    l_ts = l//k_folds

    loss_tr_mean = defaultdict(int)
    loss_ts_mean = defaultdict(int)

    for k in range(k_folds):
        # cicle over folds, for every fold create train_d, valid_d
        if k != k_folds-1: # exclude the k-th part from the validation
            train_inputs = np.append(d.text_inputs[:(k)*l_ts], d.text_inputs[(k+1)*l_ts:], axis=0)
            train_targets = np.append(d.targets[:(k)*l_ts], d.targets[(k+1)*l_ts:], axis=0)
            train_masks = np.append(d.masks[:(k)*l_ts], d.masks[(k+1)*l_ts:], axis=0)
            test_inputs = d.text_inputs[k*l_ts:(k+1)*l_ts]
            test_targets = d.targets[k*l_ts:(k+1)*l_ts]
            test_masks = d.masks[k*l_ts:(k+1)*l_ts]

        else: # last fold clausole
            train_inputs = d.text_inputs[:k*l_ts]
            train_targets = d.targets[:k*l_ts]
            train_masks = d.masks[:k*l_ts]
            test_inputs = d.text_inputs[k*l_ts:]
            test_targets = d.targets[k*l_ts:]
            test_masks = d.masks[k*l_ts:]


        LOGGER.info(f"Train data: {len(train_inputs)}")
        LOGGER.info(f"Test data: {len(test_inputs)}")

        # min max scaler the targets
        train_targets, test_targets = minMaxScaling(train_targets, test_targets, d.feature_max)

        # create the dataloader
        train_dl = GazeDataLoader(cf, train_inputs, train_targets, train_masks, d.target_pad, mode="train")
        test_dl = GazeDataLoader(cf, test_inputs, test_targets, test_masks, d.target_pad, mode="test")

        # Model
        model = load_model_from_hf(cf.model_name, not cf.random_weights, d.d_out)

        # optimizer
        optim = create_finetuning_optimizer(cf, model)

        # scheduler
        scheduler = create_scheduler(cf, optim, train_dl)

        # trainer
        trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, f"CV-Training-{k+1}/{k_folds}",
                                    DEVICE, writer=writer, test_dl=test_dl)
        trainer.train()

        for key, metric in trainer.tester.train_metrics.items():
            loss_tr_mean[key] += metric

        for key, metric in trainer.tester.test_metrics.items():
            loss_ts_mean[key] += metric

    for key in loss_tr_mean:
        loss_tr_mean[key] /= k_folds

    for key in loss_ts_mean:
        loss_ts_mean[key] /= k_folds

    return loss_tr_mean, loss_ts_mean