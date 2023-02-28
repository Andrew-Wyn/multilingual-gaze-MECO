import os
import argparse

import torch

import numpy as np
import pandas as pd

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # DataCollatorWithPadding,
    EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    Trainer,
    TrainingArguments,
    # default_data_collator,
    set_seed,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from gaze.utils import Config, LOGGER


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')
    parser.add_argument('-m', '--model-dir', dest='model_dir', action='store',
                        help=f'Relative path of finetuned model directory, containing the config and the saved weights')
    parser.add_argument('-p', '--pretrained', dest='pretrained', default=False, action='store_true',
                        help=f'Bool, start from a pretrained model')
    parser.add_argument('-f', '--finetuned', dest='finetuned', default=False, action='store_true',
                        help=f'Bool, start from a finetuned model')

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

    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    dataset_sst2 = load_dataset("sst2", cache_dir=CACHE_DIR)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding=True, truncation=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        num_train_epochs=cf.n_epochs,              # total number of training epochs
        per_device_train_batch_size=cf.train_bs,  # batch size per device during training
        per_device_eval_batch_size=cf.eval_bs,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=cf.weight_decay,               # strength of weight decay
        save_strategy="no",
        learning_rate=cf.lr
    )

    # Model
    LOGGER.info("Model retrieving...")
    LOGGER.info("Take pretrained model")

    model = AutoModelForSequenceClassification.from_pretrained(cf.model_name)

    tokenized_datasets_sst2 = dataset_sst2.map(tokenize_function, batched=True,
                                                            load_from_cache_file=CACHE_DIR)
        
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_datasets_sst2["train"],         # training dataset
        eval_dataset=tokenized_datasets_sst2["validation"],            # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()

    trainer.save_model(args.output_dir)

    # compute train results
    metrics = train_result.metrics

    # save train results
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)



    # compute evaluation results
    metrics = trainer.evaluate()

    # save evaluation results
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)