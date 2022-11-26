import argparse
import os
from collections import defaultdict
from tqdm import tqdm
import datetime
import json

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

import numpy as np

from gaze.dataset import GazeDataset
from gaze.dataloader import GazeDataLoader
from gaze.utils import LOGGER, randomize_model, Config
import torch
from transformers import (
    # AutoConfig,
    AutoModelForTokenClassification,
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

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


def load_model_from_hf(model_name, pretrained, d_out=8):
    # Model
    LOGGER.info("initiating model:")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=d_out,
                                    output_attentions=False, output_hidden_states=True)

    if not pretrained:
        # initiate Bert with random weights
        print("randomizing weights")
        model = randomize_model(model)
        #print(model.classifier.weight.data)

    return model


def compute_attention_correlation(cf, tokenizer, model):
    LOGGER.info(f"Computing Attention Correlation ...")

    modes = ["train", "valid", "test"]

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/all_mean_dataset.csv", "try")
    d.read_pipeline()

    # TODO: combine datasets
    dataset = d.numpy["train"]

    print(model.config.num_hidden_layers)

    for input, _, mask in tqdm(dataset):
        with torch.no_grad():
            model_output = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]))

        print(len(model_output.attentions))

        for attention_layer in model_output.attentions:
            print(attention_layer[0].shape)

            attention_layer_mean = torch.mean(attention_layer[0], 0)

            masked_ids = np.where(mask == 1)[0]

            print(masked_ids)

            reduced_attention_layer_mean = attention_layer_mean[masked_ids, masked_ids]

            print(reduced_attention_layer_mean.shape)

            exit(0)


def main():
    parser = argparse.ArgumentParser(description='Regression Probing')
    parser.add_argument('-m' ,'--model', dest='model_name', action='store',
                        help=f' of the model to retrieve from the HF repository')
    parser.add_argument('-d' ,'--model_dir', dest='model_dir', action='store',
                        help=f'Relative path of the pretrained model')
    parser.add_argument('-o' ,'--output_dir', dest='output_dir', action='store',
                    help=f'Relative path of the probing output')
    parser.add_argument('-l' ,'--linear', dest='linear', action=argparse.BooleanOptionalAction,
                    help=f'If apply linear model, default False')
    parser.add_argument('-a' ,'--average', dest='average', action=argparse.BooleanOptionalAction,
                    help=f'If apply average over the subtokens')
    parser.add_argument('-p' ,'--pretrained', dest='pretrained', action=argparse.BooleanOptionalAction,
                        help=f'If needed a pretrained model')
    parser.add_argument('-f' ,'--finetuned', dest='finetuned', action=argparse.BooleanOptionalAction,
                        help=f'If needed a finetuned model')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script \
                            {{ \
                                "feature_max": int, \
                            }}')


    args = parser.parse_args()

    pretrained = args.pretrained
    finetuned = args.finetuned
    model_name = args.model_name
    model_dir = args.model_dir
    output_dir = args.output_dir
    config_file = args.config_file
    linear = False if args.linear is None else True
    average = False if args.average is None else True

    cf = Config.load_json(config_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if not finetuned: # downaload from huggingface
        print("download from hf")
        model = load_model_from_hf(model_name, pretrained)

    else: # load from disk
        print("load from disk")
        model = AutoModelForTokenClassification.from_pretrained(model_dir, output_attentions=True, output_hidden_states=False)

    compute_attention_correlation(cf, tokenizer, model)

if __name__ == "__main__":
    main()