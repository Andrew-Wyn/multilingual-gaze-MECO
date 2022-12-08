from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import logging.config
import json
from sklearn.preprocessing import MinMaxScaler


CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "processing": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}


logging.config.dictConfig(CONFIG)
LOGGER = logging.getLogger("processing")


class Config:
    @classmethod
    def load_json(cls, fpath):
        cf = cls()
        with open(fpath) as f:
            cf.__dict__ = json.load(f)

        return cf


def create_finetuning_optimizer(cf, model):
    """
    Creates an Adam optimizer with weight decay. We can choose whether to perform full finetuning on
    all parameters of the model or to just optimize the parameters of the final classification layer.
    """
    if cf.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay_rate": cf.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay_rate": 0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return AdamW(optimizer_grouped_parameters, lr=cf.lr, eps=cf.eps)


def create_scheduler(cf, optim, dl):
    """
    Creates a linear learning rate scheduler.
    """
    n_iters = cf.n_epochs * len(dl)
    return get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=n_iters)


def randomize_model(model):
    #https://stackoverflow.com/questions/68058647/initialize-huggingface-bert-with-random-weights
    for module_ in model.named_modules():
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model


def mask_mse_loss(b_output, b_target, target_pad, d_out):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_mask = b_target.view(-1, d_out) == target_pad
    active_outputs = b_output.view(-1, d_out)
    active_targets = torch.where(active_mask, active_outputs, b_target.view(-1, d_out))

    return active_outputs, active_targets


class GazePredictionLoss:
    """
    Loss that deals with a list of variable length sequences. The object call returns global + per-feature MAE loss.
    """

    def __init__(self, d_gaze):
        self.d_gaze = d_gaze
        self.d_report = d_gaze + 1

        self.loss = torch.nn.L1Loss(reduction="sum")

    def __call__(self, b_output, b_target):
        b_length = [len(i) for i in b_output]
        losses = torch.zeros(self.d_report)

        losses[0] = sum([self.loss(i, j) for i, j in zip(b_output, b_target)])
        for output_orig_len, target_orig_len in zip(b_output, b_target):
            for i in range(1, self.d_report):
                losses[i] += self.loss(output_orig_len[:, i - 1], target_orig_len[:, i - 1])

        losses[0] /= sum([i * self.d_gaze for i in b_length])
        losses[1:] /= sum(b_length)
        return losses


def minMaxScaling(train_targets, test_targets=None, feature_max=1):   
    features = train_targets
    scaler = MinMaxScaler(feature_range=[0, feature_max])
    flat_features = [j for i in features for j in i]
    scaler.fit(flat_features)

    train_targets = [list(scaler.transform(i)) for i in features]
    
    if not test_targets is None:
        test_targets = [list(scaler.transform(i)) for i in test_targets]
        return train_targets, test_targets
    
    return train_targets