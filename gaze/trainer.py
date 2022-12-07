import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from gaze.early_stopping import GazeEarlyStopping
from gaze.utils import LOGGER, mask_mse_loss
from sklearn.utils import shuffle
from gaze.utils import LOGGER, create_finetuning_optimizer, create_scheduler, randomize_model, Config
from modeling.custom_bert import BertForTokenClassification
from gaze.dataloader import GazeDataLoader
from sklearn.preprocessing import MinMaxScaler


class Trainer(ABC):
    def __init__(self, cf, model, train_dl, eval_dir, early_stop, task, device, writer):
        self.model = model
        self.train_dl = train_dl
        self.eval_dir = eval_dir
        self.early_stop = early_stop
        self.n_epochs = cf.n_epochs
        self.task = task
        self.device = device
        self.writer = writer

    @abstractmethod
    def train_one_step(self, batch):
        pass

    def train(self):
        n_batches_one_epoch = len(self.train_dl)
        n_params = sum(p.numel() for p in self.model.parameters())
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
                self.writer.add_scalar("train/loss", loss, it)
                epoch_loss_ls.append(loss)

            epoch_loss_avg = sum(epoch_loss_ls) / len(epoch_loss_ls)
            epoch_loss_ls = []
            LOGGER.info(f"Done epoch {epoch} / {self.n_epochs}")
            LOGGER.info(f"Avg loss epoch {epoch}: {epoch_loss_avg:.4f}")

            # self.early_stop()

            for key, metric in self.early_stop.tester.metrics.items():
               self.writer.add_scalar(f"val/{key}", metric, it // n_batches_one_epoch)

            # if self.early_stop.stop:
            #     break


class GazeTrainer(Trainer):
    def __init__(self, cf, model, train_dl, val_dl, optim, scheduler, eval_dir,
                 task, device, monitor, monitor_mode, writer):
        early_stop = GazeEarlyStopping(cf, model, val_dl, eval_dir, device, task, monitor, monitor_mode, writer)
        super().__init__(cf, model, train_dl, eval_dir, early_stop, task, device, writer)

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


def cross_validation(cf, d, eval_dir, writer, DEVICE, k_folds=10):
    """
    Perform a k-fold cross-validation
    """

    l = len(d.text_inputs)
    l_ts = l//k_folds

    loss_tr_mean = 0
    loss_ts_mean = 0

    # shuffle dataset with a seed, to take a reproducible sample
    # dataset = shuffle(dataset, random_state=42)

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

        # min max scaler the targets
        features = train_targets
        scaler = MinMaxScaler(feature_range=[0, d.feature_max])
        flat_features = [j for i in features for j in i]
        scaler.fit(flat_features)

        train_targets = [list(scaler.transform(i)) for i in features]
        test_targets = [list(scaler.transform(i)) for i in test_targets]

        # create the dataset
        train_d = list(zip(train_inputs, train_targets, train_masks))
        test_d = list(zip(test_inputs, test_targets, test_masks))

        train_dl = GazeDataLoader(cf, train_d, d.target_pad, mode="train")
        test_dl = GazeDataLoader(cf, test_d, d.target_pad, mode="test")

        # Model
        LOGGER.info("initiating model: ")
        model = BertForTokenClassification.from_pretrained(cf.model_pretrained, num_labels=d.d_out,
                                        output_attentions=False, output_hidden_states=False)

        if cf.random_weights is True:
            # initiate Bert with random weights
            print("randomizing weights")
            model = randomize_model(model)
            #print(model.classifier.weight.data)

        # optimizer
        optim = create_finetuning_optimizer(cf, model)

        # scheduler
        scheduler = create_scheduler(cf, optim, train_dl)

        # trainer
        trainer = GazeTrainer(cf, model, train_dl, test_dl, optim, scheduler, eval_dir, "task",
                                    DEVICE, monitor="loss_all", monitor_mode="min", writer=writer)
        trainer.train()
        LOGGER.info(f"Training completed task")

        # loss_tr_mean += history["loss_tr"]
        # loss_ts_mean += history["loss_vl"]

    loss_tr_mean /= k_folds
    loss_ts_mean /= k_folds

    return loss_tr_mean, loss_ts_mean