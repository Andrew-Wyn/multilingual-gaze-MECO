import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model import GazePredictionLoss
from utils import mask_mse_loss, LOGGER
import os
from abc import ABC, abstractmethod

class Tester(ABC):
    def __init__(self, quantities, device, task):
        self.quantities = quantities  # list of metrics to be evaluated (other than the loss)
        self.device = device
        self.task = task

        self.preds = []
        self.logs = []
        self.maes = []
        self.metrics = {}  # key-value dictionary metric --> value
        self.units = {}  # key-value dictionary metric --> measurement unit

    def evaluate(self):
        LOGGER.info(f"Begin evaluation task {self.task}")
        self.predict()

        LOGGER.info("Calulating metrics")
        self.calc_metrics()

        for key in self.metrics:
            LOGGER.info(f"val_{key}: {self.metrics[key]:.4f} {self.units[key]}")

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def calc_metrics(self):
        pass

    @abstractmethod
    def save_preds(self, fpath):
        pass

    def save_logs(self, fpath):
        dir = os.path.dirname(fpath)
        if not os.path.exists(dir):
            os.makedirs(dir)

        for key in self.metrics:
            self.logs.append(f"{key}: {self.metrics[key]:.4f} {self.units[key]}\n")

        with open(fpath, "w") as f:
            f.writelines(self.logs)

    def save_logs_all(self, fpath, seed, config):
        dir = os.path.dirname(fpath)
        if not os.path.exists(dir):
            os.makedirs(dir)

        for key in self.metrics:
            self.maes.append(self.metrics[key])

        log_text = [self.task] + self.maes[1:] + [seed, config.model_pretrained, config.full_finetuning, config.random_weights, config.random_baseline, "\n"]

        with open(fpath, "a") as f:
            f.write("\t".join(map(str, log_text)))



class GazeTester(Tester):
    def __init__(self, model, dl, device, task):
        quantities = []
        super().__init__(quantities, device, task)

        self.model = model
        self.dl = dl
        self.target_pad = dl.target_pad

        self.criterion = nn.MSELoss(reduction="mean")
        self.criterion_metric = GazePredictionLoss(model.d_out)

    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            loss = 0
            losses_metric = torch.zeros(self.criterion_metric.d_report)
            self.preds = []

            for batch in tqdm(self.dl):
                b_input, b_target, b_mask = batch
                b_input = b_input.to(self.device)
                b_target = b_target.to(self.device)
                b_mask = b_mask.to(self.device)

                b_output = self.model(input_ids=b_input, attention_mask=b_mask)[0]

                active_outputs, active_targets = mask_mse_loss(b_output, b_target, self.target_pad, self.model.d_out)
                loss += self.criterion(active_outputs, active_targets)

                b_output_orig_len = []
                b_target_orig_len = []
                for output, target in zip(b_output, b_target):
                    active_idxs = (target != self.target_pad)[:, 0]
                    b_output_orig_len.append(output[active_idxs])
                    b_target_orig_len.append(target[active_idxs])

                losses_metric += self.criterion_metric(b_output_orig_len, b_target_orig_len)

                self.preds.extend([i.cpu().numpy() for i in b_output_orig_len])

            num_batches = len(self.dl)
            loss /= num_batches
            losses_metric /= num_batches

            self.metrics["loss"] = loss.item()
            self.units["loss"] = ""
            self.metrics["loss_all"] = losses_metric[0].item()
            self.units["loss_all"] = ""
            for i, value in enumerate(losses_metric[1:]):
                self.metrics["loss_" + str(i)] = value.item()
                self.units["loss_" + str(i)] = ""

    def calc_metrics(self):
        pass

    def save_preds(self, fpath):
        dir = os.path.dirname(fpath)
        if not os.path.exists(dir):
            os.makedirs(dir)

        flat_preds = [j for i in self.preds for j in i]

        preds_pd = pd.DataFrame(flat_preds, columns=["n_fix", "first_fix_dur", "first_pass_dur",
                                                     "total_fix_dur", "mean_fix_dur", "fix_prob",
                                                     "n_refix", "reread_prob"])
        preds_pd.to_csv(fpath)