import os
from abc import ABC
from utils import LOGGER
from tester import GazeTester


class EarlyStopping(ABC):
    def __init__(self, cf, model, dir, monitor, monitor_mode, tester):
        self.model = model
        self.dir = dir
        self.patience = cf.patience
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.tester = tester

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
            #for key, value in self.tester.metrics.items():
                # mlflow.log_metric(f"val_{key}", value)

            self.best_score = score

            LOGGER.info("Metric has improved, saving the model")
            # torch.save(self.model.state_dict(), os.path.join(self.dir, "model-"+cf.model_pretrained+"-"+str(cf.full_finetuning)+"-"+str(RANDOM_STATE)+".pth"))

            self.run_patience = 0
        else:
            self.run_patience += 1
            LOGGER.info(f"No improvement in the last epoch, patience {self.run_patience} out of {self.patience}")


class GazeEarlyStopping(EarlyStopping):
    def __init__(self, cf, model, val_dataloader, dir, device, task, monitor, monitor_mode):
        tester = GazeTester(model, val_dataloader, device, task)
        super().__init__(cf, model, dir, monitor, monitor_mode, tester)