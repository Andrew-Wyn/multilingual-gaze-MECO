from gaze.dataset import GazeDataset
from gaze.utils import Config
from gaze.trainer import cross_validation
import torch
from transformers import (
    # AutoConfig,
    # AutoModelForTokenClassification,
    AutoTokenizer,
    # DataCollatorWithPadding,
    # EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    # set_seed,
)


import os
from torch.utils.tensorboard import SummaryWriter
import argparse


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script \
                            {{ \
                                "feature_max": int, \
                                "model_pretrained": str, \
                                "finetune_on_gaze": boolean, \
                                "full_finetuning": boolean, \
                                "weight_decay": float, \
                                "lr": float, \
                                "eps": float, \
                                "max_grad_norm": float, \
                                "train_bs": int, \
                                "eval_bs": int, \
                                "n_epochs": int, \
                                "patience": int, \
                                "random_weights": boolean \
                            }}')

    args = parser.parse_args()
    config_file = args.config_file

    cf = Config.load_json(config_file)

    eval_dir = "eval_dir"

    print(cf)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    writer = SummaryWriter(eval_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_pretrained, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/it/all_mean_dataset.csv", "try")
    d.read_pipeline()

    cross_validation(cf, d, eval_dir, writer, DEVICE, k_folds=10)

if __name__ == "__main__":
    main()