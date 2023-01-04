import argparse
import os
from collections import defaultdict
from tqdm import tqdm
import datetime
import json

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


import numpy as np

from gaze.dataset import GazeDataset
from gaze.utils import LOGGER, Config
import torch
from transformers import (
    AutoConfig,
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


def create_probing_dataset(d, model, mean=False):
    LOGGER.info(f"Creating datasets, Mean = {mean} ...")

    LOGGER.info("Splitting dataset in train and test ...")

    probing_dataset = defaultdict(list)

    print(model.config.num_hidden_layers)

    LOGGER.info(f"Start creating dataset...")

    for input, target, mask in tqdm(zip(d.text_inputs, d.targets, d.masks)):
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

            # take elements token-wise
            for i in range(input.shape[0]):
                probing_dataset[layer].append((input[i], output[i]))
            
    LOGGER.info("Retrieving done, postprocess...")
    
    # concatenate the inputs and outputs !!!!
    for layer in range(model.config.num_hidden_layers):
        input_list = []
        output_list = []
        
        for input, output in probing_dataset[layer]:
            input_list.append(input)
            output_list.append(output)

        probing_dataset[layer] = (input_list, output_list)
    
    return probing_dataset


def apply_model(inputs, targets, feature_max, linear = True, k_folds=10, num_labels=8):
    # do cross-validation

    l = len(inputs)
    l_ts = l//k_folds

    loss_tr_mean = np.zeros(num_labels + 1)
    loss_ts_mean = np.zeros(num_labels + 1)

    for k in tqdm(range(k_folds)):
        # cicle over folds, for every fold create train_d, valid_d
        if k != k_folds-1: # exclude the k-th part from the validation
            train_inputs = inputs[:(k)*l_ts] + inputs[(k+1)*l_ts:]
            train_targets = targets[:(k)*l_ts] + targets[(k+1)*l_ts:]
            test_inputs = inputs[k*l_ts:(k+1)*l_ts]
            test_targets = targets[k*l_ts:(k+1)*l_ts]

        else: # last fold clausole
            train_inputs = inputs[:k*l_ts]
            train_targets = targets[:k*l_ts]
            test_inputs = inputs[k*l_ts:]
            test_targets = targets[k*l_ts:]

        # min max scaler the targets
        scaler = MinMaxScaler(feature_range=[0, feature_max])
        flat_features = [i for i in train_targets]
        scaler.fit(flat_features)
        train_targets = scaler.transform(train_targets)
        test_targets = scaler.transform(test_targets)

        if linear:
            regr = MultiOutputRegressor(SVR()).fit(train_inputs, train_targets)
        else:
            regr = MLPRegressor().fit(train_inputs, train_targets)

        predicted_train = regr.predict(train_inputs)
        predicted_test = regr.predict(test_inputs)

        loss_tr_mean += np.concatenate((([mean_absolute_error(train_targets, predicted_train)], mean_absolute_error(train_targets, predicted_train, multioutput='raw_values'))), axis=0)
        loss_ts_mean += np.concatenate(([mean_absolute_error(train_targets, predicted_train)], mean_absolute_error(test_targets, predicted_test, multioutput='raw_values')), axis=0)

    loss_tr_mean /= k_folds
    loss_ts_mean /= k_folds

    return loss_tr_mean, loss_ts_mean


def probe(probing_dataset, feature_max, linear, output_dir, k_folds, num_labels):
    LOGGER.info(f"Starting probe, Linear = {linear} ...")
    metrics = dict()

    metrics["linear"] = linear

    for layer, dataset in probing_dataset.items():
        LOGGER.info(f"Cross Validation layer : {layer} ...")

        inputs, targets = dataset

        score_train, score_test = apply_model(inputs, targets, feature_max, linear, k_folds, num_labels)

        metrics[layer] = {
            "score_train" : score_train,
            "score_test" : score_test
        }

        metrics[layer]["score_train"] = score_train.tolist()
        metrics[layer]["score_test"] = score_test.tolist()

        LOGGER.info(f"Scores layer - {layer} :")
        LOGGER.info(f"Train: {score_train.tolist()}")
        LOGGER.info(f"Test: {score_test.tolist()}")
        LOGGER.info(f"done!!!")

        print(metrics)

    with open(f"{output_dir}/probe_results_{datetime.datetime.now().time()}.json", 'w') as f:
        json.dump(metrics, f)


def load_model_from_hf(model_name, pretrained, d_out=8):
    # Model
    LOGGER.info("Initiating model ...")
    if not pretrained:
        # initiate model with random weights
        LOGGER.info("Take randomized model:")
        config = AutoConfig.from_pretrained(model_name, num_labels=d_out,
                                    output_attentions=False, output_hidden_states=True)
        model = AutoModelForTokenClassification.from_config(config)
    else:
        LOGGER.info("Take pretrained model:")
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=d_out,
                            output_attentions=False, output_hidden_states=True)

    return model


def main():
    parser = argparse.ArgumentParser(description='Regression Probing')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')

    args = parser.parse_args()

    config_file = args.config_file

    cf = Config.load_json(config_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    # TODO: LOAD DATASET BEFORE MODEL CREATION
    # Dataset
    d = GazeDataset(cf, tokenizer, cf.dataset)
    d.read_pipeline()
    d.randomize_data()

    if not cf.finetuned: # downaload from huggingface
        LOGGER.info("Model retrieving, download from hf...")
        model = load_model_from_hf(cf.model_name, cf.pretrained, d.d_out)

    else: # load from disk
        LOGGER.info("Model retrieving, load from disk...")
        model = AutoModelForTokenClassification.from_pretrained(cf.model_dir, output_attentions=False, output_hidden_states=True)
    
    LOGGER.info("Model retrieved !")

    probing_dataset = create_probing_dataset(d, model, mean=cf.average)
    probe(probing_dataset, cf.feature_max, cf.linear, cf.output_dir, cf.k_fold, model.num_labels)


if __name__ == "__main__":
    main()