import os
from abc import ABC
from gaze.utils import LOGGER
from gaze.tester import GazeTester
import torch

class EarlyStopping(ABC):
    def __init__(self, cf, model, dir, monitor, monitor_mode, tester, writer):
        self.cf = cf
        self.save_counter = 0
        self.model = model
        self.dir = dir
        self.patience = cf.patience
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.tester = tester
        self.writer = writer

        self.run_patience = 0
        self.best_score = None
        self.stop = False

    def __call__(self):
        if self.run_patience == self.patience:
            LOGGER.info("Patience exceeded, stopping")
            self.stop = True
            return

        self.tester.evaluate()
        score = self.tester.metrics[self.monitor]

        self.model.train()

        if self.best_score is None or (self.monitor_mode == "min" and score < self.best_score) or \
                (self.monitor_mode == "max" and score > self.best_score):
            for key, value in self.tester.metrics.items():
                self.writer.add_scalar(f"best_score/val/{key}", value, self.save_counter)

            self.best_score = score

            LOGGER.info("Metric has improved, saving the model")
            # torch.save(self.model.state_dict(), os.path.join(self.dir, "model-"+self.cf.model_pretrained+"-"+str(self.cf.full_finetuning)+"-"+str(self.save_counter)+".pth"))
            # need to save with this utility to load in a second step
            self.model.save_pretrained(os.path.join(self.dir, "model-"+self.cf.model_pretrained+"-"+str(self.cf.full_finetuning)+"-"+str(self.save_counter)))
            self.save_counter += 1

            self.run_patience = 0
        else:
            self.run_patience += 1
            LOGGER.info(f"No improvement in the last epoch, patience {self.run_patience} out of {self.patience}")


class GazeEarlyStopping(EarlyStopping):
    def __init__(self, cf, model, val_dataloader, dir, device, task, monitor, monitor_mode, writer):
        tester = GazeTester(model, val_dataloader, device, task)
        super().__init__(cf, model, dir, monitor, monitor_mode, tester, writer)