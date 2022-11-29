import argparse
import os
from tqdm import tqdm

import scipy

from scipy.stats import spearmanr

import numpy as np

from gaze.dataset import GazeDataset
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


def create_attention_dataset(cf, tokenizer, model, subwords=False):
    LOGGER.info(f"Creating Attention Dataset, subwords sum : {subwords}...")

    modes = ["train", "valid", "test"]

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/all_mean_dataset.csv", "try")
    d.read_pipeline()

    dataset = []

    for mode in modes:
        dataset += d.numpy[mode]
    
    attention_list_layers = list()
    target_list_layers = list()

    print(model.config.num_hidden_layers)

    for input, target, mask in tqdm(dataset):
        with torch.no_grad():
            model_output = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]))

        attention_dict = dict()
        target_dict = dict()

        for layer, attention_layer in enumerate(model_output.attentions):
            # We use the first element of the batch because batch size is 1
            attention = attention_layer[0].numpy()

            non_padded_ids = np.where(mask == 1)[0]

            # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
            # I also tried the sum once, but the result was even worse
            mean_attention = np.mean(attention, axis=0)

            # We drop padded tokens
            # mean_attention = mean_attention[non_padded_ids]

            # We drop CLS and SEP tokens
            mean_attention = mean_attention[1:-1]

            # 2. For each word, we sum over the attention to the other words to determine relative importance
            sum_attention = np.sum(mean_attention, axis=0)

            sum_attention = sum_attention[non_padded_ids]

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

            # Taking the softmax does not make a difference for calculating correlation
            # It can be useful to scale the salience signal to the same range as the human attention
            relative_attention = scipy.special.softmax(sum_attention)

            attention_dict[layer] = relative_attention

            word_targets = target[first_token_ids]

            # normalize features between [0,1]
            sum_features = np.sum(word_targets, axis=0)

            normalized_features = word_targets / sum_features

            # avoid division by 0 in the normalization of a full zero row
            normalized_features[np.isnan(normalized_features)] = 0


            target_dict[layer] = list()

            for feat_i in range(normalized_features.shape[1]):
                target_dict[layer].append(normalized_features[:, feat_i])
                

        attention_list_layers.append(attention_dict)
        target_list_layers.append(target_dict)

    # transform datasets, from list of dicts to dict of lists
    return_dict_attns = dict()
    return_dict_target = dict()
    for layer in range(model.config.num_hidden_layers):
        return_dict_attns[layer] = list()
        return_dict_target[layer] = list()
    
    for attention_dict, target_dict in zip(attention_list_layers, target_list_layers):
        for layer in range(model.config.num_hidden_layers):
            return_dict_attns[layer].append(attention_dict[layer])
            return_dict_target[layer].append(target_dict[layer])

    return return_dict_attns, return_dict_target


class AttentionRollout():
    def compute_flows(self, attentions_list, desc="", output_hidden_states=False, disable_tqdm=False):
        """
        :param attentions_list: list of attention maps (#examples, #layers, #sent_len, #sent_len)
        :param desc:
        :param output_hidden_states:
        :param num_cpus:
        :return:
        """
        attentions_rollouts = []
        for i in tqdm(range(len(attentions_list)), desc=desc, disable=disable_tqdm):
            if output_hidden_states:
                attentions_rollouts.append(self.compute_joint_attention(attentions_list[i]))
            else:
                attentions_rollouts.append(self.compute_joint_attention(attentions_list[i])[[-1]])
        return attentions_rollouts

    def compute_joint_attention(self, att_mat):
        res_att_mat = att_mat
        # res_att_mat = res_att_mat[4:10, :, :]
        joint_attentions = np.zeros(res_att_mat.shape)
        layers = joint_attentions.shape[0]
        joint_attentions[0] = res_att_mat[0]
        for i in np.arange(1, layers):
            joint_attentions[i] = res_att_mat[i].dot(joint_attentions[i - 1])

        return joint_attentions


def create_globenc_dataset(cf, tokenizer, model, subwords=False):
    LOGGER.info(f"Creating Attention Dataset, subwords sum : {subwords}...")

    modes = ["train", "valid", "test"]

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/all_mean_dataset.csv", "try")
    d.read_pipeline()

    dataset = []

    for mode in modes:
        dataset += d.numpy[mode]
    
    attention_list_layers = list()
    target_list_layers = list()

    print(model.config.num_hidden_layers)

    for input, target, mask in tqdm(dataset):
        # demo GLOBENC
        with torch.no_grad():
            logits, attentions, norms = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]), output_norms=True, return_dict=False)

        num_layers = len(attentions)
        norm_nenc = torch.stack([norms[i][4] for i in range(num_layers)]).squeeze().cpu().numpy()
        print("Single layer N-Enc token attribution:", norm_nenc.shape)

        # Aggregate and compute GlobEnc
        globenc = AttentionRollout().compute_flows([norm_nenc], output_hidden_states=True)[0]
        globenc = np.array(globenc)
        print("Aggregated N-Enc token attribution (GlobEnc):", globenc.shape)

        exit(0)

        attention_dict = dict()
        target_dict = dict()

        for layer, attention_layer in enumerate(model_output.attentions):
            # We use the first element of the batch because batch size is 1
            attention = attention_layer[0].numpy()

            non_padded_ids = np.where(mask == 1)[0]

            # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
            # I also tried the sum once, but the result was even worse
            mean_attention = np.mean(attention, axis=0)

            # We drop padded tokens
            # mean_attention = mean_attention[non_padded_ids]

            # We drop CLS and SEP tokens
            mean_attention = mean_attention[1:-1]

            # 2. For each word, we sum over the attention to the other words to determine relative importance
            sum_attention = np.sum(mean_attention, axis=0)

            sum_attention = sum_attention[non_padded_ids]

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

            # Taking the softmax does not make a difference for calculating correlation
            # It can be useful to scale the salience signal to the same range as the human attention
            relative_attention = scipy.special.softmax(sum_attention)

            attention_dict[layer] = relative_attention

            word_targets = target[first_token_ids]

            # normalize features between [0,1]
            sum_features = np.sum(word_targets, axis=0)

            normalized_features = word_targets / sum_features

            # avoid division by 0 in the normalization of a full zero row
            normalized_features[np.isnan(normalized_features)] = 0


            target_dict[layer] = list()

            for feat_i in range(normalized_features.shape[1]):
                target_dict[layer].append(normalized_features[:, feat_i])
                

        attention_list_layers.append(attention_dict)
        target_list_layers.append(target_dict)

    # transform datasets, from list of dicts to dict of lists
    return_dict_attns = dict()
    return_dict_target = dict()
    for layer in range(model.config.num_hidden_layers):
        return_dict_attns[layer] = list()
        return_dict_target[layer] = list()
    
    for attention_dict, target_dict in zip(attention_list_layers, target_list_layers):
        for layer in range(model.config.num_hidden_layers):
            return_dict_attns[layer].append(attention_dict[layer])
            return_dict_target[layer].append(target_dict[layer])

    return return_dict_attns, return_dict_target


def compute_spearman_correlation(attns, targets):
    spearman_correlations = list()

    for attn, targ in zip(attns, targets):
        spearman_correlations.append(spearmanr(attn, targ)[0])

    return np.mean(spearman_correlations)


def compute_correlations(dict_attns, dict_target, num_features=8, num_layers=6):
    correlations = dict()

    for feat in range(num_features):
        correlations[feat] = dict()
        for layer in range(num_layers):
            layer_target_feature = [targets[feat] for targets in dict_target[layer]]
            layer_attns = dict_attns[layer]
            correlations[feat][layer] = compute_spearman_correlation(layer_attns, layer_target_feature)

    return correlations

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

    cf = Config.load_json(config_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if not finetuned: # downaload from huggingface
        print("download from hf")
        model = load_model_from_hf(model_name, pretrained)

    else: # load from disk
        print("load from disk")
        model = BertForTokenClassification.from_pretrained(model_dir, output_attentions=True, output_hidden_states=False)

    dict_attns, dict_target = create_globenc_dataset(cf, tokenizer, model)

    correlations = compute_correlations(dict_attns, dict_target)

    print(correlations)

if __name__ == "__main__":
    main()