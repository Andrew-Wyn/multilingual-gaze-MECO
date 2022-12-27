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

from modeling.custom_bert import BertForTokenClassification

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


def create_probing_dataset(cf, tokenizer, model, mean=False):
    LOGGER.info(f"Creating datasets, Mean = {mean} ...")
    probing_dataset = dict()

    modes = ["train", "valid", "test"]

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/all_mean_dataset.csv", "try")
    d.read_pipeline()

    # train_dl = GazeDataLoader(cf, d.numpy["train"], d.target_pad, mode="train")
    # val_dl = GazeDataLoader(cf, d.numpy["valid"], d.target_pad, mode="val") 

    print(model.config.num_hidden_layers)

    for mode in modes:

        LOGGER.info(f"Start creating - {mode} - dataset...")

        probing_dataset[mode] = defaultdict(list)

        for input, target, mask in tqdm(d.numpy[mode]):
            # print(input)
            # print(mask)
            # print(np.multiply.reduce(target != -1, 1) > 0)

            last_token_id = (len(mask) - 1 - mask.tolist()[::-1].index(1)) - 1

            # remove the tokens with -1 in the target

            with torch.no_grad():
                model_output = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]))
            
            for layer in range(model.config.num_hidden_layers):

                hidden_state = model_output.hidden_states[layer].numpy()

                non_masked_els = np.multiply.reduce(target != -1, 1) > 0

                if not mean:
                    # take only the first subword embedding for a given word
                    input = hidden_state[0, non_masked_els, :]
                else:
                    # take the mean of the subwords's embedding for a given word
                    # id of the words's start
                    input = [np.mean(split_, 0) for split_ in np.split(hidden_state[0], np.where(non_masked_els)[0], 0)[1:-1]]
                    # the last have mean all the vector from the last non masked to the sep token (sep token is the last 1 in mask)
                    last_mean = np.mean(hidden_state[0, np.where(non_masked_els)[0][-1] : last_token_id, :], 0)
                    input.append(last_mean)

                    input = np.array(input)

                output = target[non_masked_els, :]

                probing_dataset[mode][layer].append((input, output))
            
        LOGGER.info("Retrieving done, postprocess...")
        
        # concatenate the inputs and outputs !!!!
        for layer in range(model.config.num_hidden_layers):
            input_list = []
            output_list = []
            
            for input, output in probing_dataset[mode][layer]:
                input_list.append(input)
                output_list.append(output)

            input_list = np.concatenate(input_list, axis=0)
            output_list = np.concatenate(output_list, axis=0)

            probing_dataset[mode][layer] = (input_list, output_list)

        LOGGER.info(f"{mode} done!")

    # transform dataset
    return_probing_dataset = dict()

    for layer in range(model.config.num_hidden_layers):
        return_probing_dataset[layer] = dict()

        for mode in modes:
            return_probing_dataset[layer][mode] = probing_dataset[mode][layer]
    
    return return_probing_dataset


def apply_linear_model(datasets):
    input_list_train, output_list_train = datasets["train"]
    input_list_valid, output_list_valid = datasets["valid"]
    input_list_test, output_list_test = datasets["test"]
    regr = MultiOutputRegressor(SVR()).fit(input_list_train, output_list_train)
    predicted_train = regr.predict(input_list_train)
    predicted_valid = regr.predict(input_list_valid)
    predicted_test = regr.predict(input_list_test)
    return r2_score(predicted_train, output_list_train), r2_score(predicted_valid, output_list_valid), r2_score(predicted_test, output_list_test)


def apply_nonlinear_model(datasets):
    input_list_train, output_list_train = datasets["train"]
    input_list_valid, output_list_valid = datasets["valid"]
    input_list_test, output_list_test = datasets["test"]
    regr = MLPRegressor().fit(input_list_train, output_list_train)
    predicted_train = regr.predict(input_list_train)
    predicted_valid = regr.predict(input_list_valid)
    predicted_test = regr.predict(input_list_test)
    return r2_score(predicted_train, output_list_train), r2_score(predicted_valid, output_list_valid), r2_score(predicted_test, output_list_test)



def probe(probing_dataset, linear, output_dir):
    LOGGER.info(f"Starting probe, Linear = {linear} ...")
    metrics = dict()

    metrics["linear"] = linear

    for layer, datasets in probing_dataset.items():
        LOGGER.info(f"---- {layer} ----")
        if linear:
            score_train, score_valid, score_test = apply_linear_model(datasets)
        else:
            score_train, score_valid, score_test = apply_nonlinear_model(datasets)

        metrics[layer] = {
            "score_train" : score_train,
            "score_valid" : score_valid,
            "score_test" : score_test
        }

        LOGGER.info("Scores:")
        LOGGER.info(f"Train: {score_train}")
        LOGGER.info(f"Valid: {score_valid}")
        LOGGER.info(f"Test: {score_test}")
        LOGGER.info(f"{layer} done!!!")

    with open(f"{output_dir}/probe_results_{datetime.datetime.now().time()}.json", 'w') as f:
        json.dump(metrics, f)


def load_model_from_hf(model_name, pretrained, d_out=8):
    # Model
    LOGGER.info("initiating model:")
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=d_out,
                                    output_attentions=False, output_hidden_states=True)

    if not pretrained:
        # initiate Bert with random weights
        print("randomizing weights")
        model = randomize_model(model)
        #print(model.classifier.weight.data)

    return model


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
        model = BertForTokenClassification.from_pretrained(model_dir, output_attentions=False, output_hidden_states=True)
    
    probing_dataset = create_probing_dataset(cf, tokenizer, model, mean=average)
    probe(probing_dataset, linear, output_dir)

if __name__ == "__main__":
    main()