import argparse
import os
from tqdm import tqdm

import scipy

import numpy as np

from gaze.dataset import GazeDataset
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


def compute_attention_correlation(cf, tokenizer, model, subwords=True):
    LOGGER.info(f"Computing Attention Correlation ...")

    modes = ["train", "valid", "test"]

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/all_mean_dataset.csv", "try")
    d.read_pipeline()

    dataset = []

    for mode in modes:
        dataset += d.numpy[mode]
    
    attention_list_layers = dict()
    target_list_layers = dict()

    print(model.config.num_hidden_layers)

    for input, target, mask in tqdm(dataset):
        with torch.no_grad():
            model_output = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]))

        for attention_layer in model_output.attentions:
            # We use the first element of the batch because batch size is 1
            attention = attention_layer[0].numpy()

            non_padded_ids = np.where(mask == 1)[0]

            # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
            # I also tried the sum once, but the result was even worse
            mean_attention = np.mean(attention, axis=0)

            print(mean_attention.shape)

            # We drop padded tokens
            # mean_attention = mean_attention[non_padded_ids]

            # We drop CLS and SEP tokens
            mean_attention = mean_attention[1:-1]

            # 2. For each word, we sum over the attention to the other words to determine relative importance
            sum_attention = np.sum(mean_attention, axis=0)

            sum_attention = sum_attention[non_padded_ids]

            print(sum_attention.shape)

            first_token_ids = np.where(np.multiply.reduce(target[non_padded_ids] != -1, 1) > 0)[0]

            # merge subwords 
            if not subwords:
                # take the attention of only the first token of a word
                sum_attention = sum_attention[first_token_ids]
            else:
                # take the sum of the subwords's attention for a given word
                # id of the words's start
                attns = [np.sum(split_) for split_ in np.split(sum_attention, first_token_ids, 0)[1:-1]]
                # the last have sum all the attention from the last non masked to the sep token (sep token is the last 1 in mask)
                last_sum = np.sum(sum_attention[first_token_ids[-1] : non_padded_ids[-1]], 0)
                attns.append(last_sum)

                sum_attention = np.array(attns)

            print(sum_attention.shape)

            # Taking the softmax does not make a difference for calculating correlation
            # It can be useful to scale the salience signal to the same range as the human attention
            relative_attention = scipy.special.softmax(sum_attention)

            print(relative_attention.shape)

            word_targets = target[first_token_ids]

            # normalize features between [0,1]

            sum_features = np.sum(word_targets, axis=0)

            normalized_features = word_targets / sum_features

            for feat_i in range(normalized_features.shape[1]):
                pass

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