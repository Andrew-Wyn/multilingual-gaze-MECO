from gaze.dataset import GazeDataset
from gaze.utils import Config
from gaze.trainer import cross_validation
import torch
from collections import defaultdict
import numpy as np
import json
from gaze.utils import LOGGER, create_finetuning_optimizer, create_scheduler, Config, minMaxScaling
from modeling.custom_bert import BertForTokenClassification
from gaze.dataloader import GazeDataLoader
from sklearn.preprocessing import MinMaxScaler
from gaze.trainer import GazeTrainer
from transformers import (
    AutoConfig,
    BertForTokenClassification,
    AutoTokenizer,
    # DataCollatorWithPadding,
    # EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    set_seed,
)

from modeling.model_resbert import ResbertConfig, ResbertForTokenClassification

import os
from torch.utils.tensorboard import SummaryWriter
import argparse


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_model(model_dir, reservoir, pretrained, d_out):
    # Model
    LOGGER.info("Initiating model ...")

    if reservoir:
        model = ResbertForTokenClassification.from_pretrained(model_dir, num_labels=d_out,
                                    output_attentions=False, output_hidden_states=True)
    else:
        model = BertForTokenClassification.from_pretrained(model_dir, num_labels=d_out,
                                    output_attentions=False, output_hidden_states=True)

    if not pretrained:
        model = randomize_model(model)

    return model


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
        model = load_model(cf.model_dir, cf.reservoir, not cf.random_weights, d.d_out)

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


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')

    # Read the script's argumenents
    args = parser.parse_args()
    config_file = args.config_file

    # Load the .json configuration file
    cf = Config.load_json(config_file)

    # set seed
    set_seed(cf.seed)

    # check if the output directory exists, if not create it!
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Writer
    writer = SummaryWriter(args.output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, args.dataset)
    d.read_pipeline()
    d.randomize_data()

    # K-fold cross-validation
    train_losses, test_losses = cross_validation(cf, d, writer, DEVICE, k_folds=cf.k_folds)

    print("Train averaged losses:")
    print(train_losses)

    print("Test averaged losses:")
    print(test_losses)

    # Retrain over all dataset

    # min max scaler the targets
    d.targets = minMaxScaling(d.targets, feature_max=d.feature_max, pad_token=d.target_pad)

    # create the dataloader
    train_dl = GazeDataLoader(cf, d.text_inputs, d.targets, d.masks, d.target_pad, mode="train")

    # Model
    model = load_model(cf.model_dir, cf.reservoir, not cf.random_weights, d.d_out)

    # Optimizer
    optim = create_finetuning_optimizer(cf, model)

    # Scheduler
    scheduler = create_scheduler(cf, optim, train_dl)

    # Trainer
    trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, f"Final_Training",
                                DEVICE, writer=writer)
    trainer.train(save_model=True, output_dir=args.output_dir)

    loss_tr = dict()

    for key, metric in trainer.tester.train_metrics.items():
        loss_tr[key] = metric

    with open(f"{args.output_dir}/finetuning_results.json", 'w') as f:
        json.dump({"losses_tr" : train_losses, "losses_ts" : test_losses, "final_training" : loss_tr}, f)

if __name__ == "__main__":
    main()