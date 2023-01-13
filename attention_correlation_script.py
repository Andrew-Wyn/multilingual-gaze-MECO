import argparse
import datetime
import json
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

from modeling.custom_xlm_roberta import XLMRobertaForTokenClassification
from modeling.custom_bert import BertForTokenClassification

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


def load_model_from_hf(tokenClassifier, model_name, pretrained, d_out=8):
    # Model
    LOGGER.info("initiating model:")
    model = tokenClassifier.from_pretrained(model_name, num_labels=d_out,
                                    output_attentions=True, output_hidden_states=False)

    if not pretrained:
        # initiate Bert with random weights
        print("randomizing weights")
        model = randomize_model(model)
        #print(model.classifier.weight.data)

    return model


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
        for i in range(len(attentions_list)):
            if output_hidden_states:
                attentions_rollouts.append(self.compute_joint_attention(attentions_list[i]))
            else:
                attentions_rollouts.append(self.compute_joint_attention(attentions_list[i])[[-1]])
        return attentions_rollouts

    def compute_joint_attention(self, att_mat):
        res_att_mat = att_mat
        joint_attentions = np.zeros(res_att_mat.shape)
        layers = joint_attentions.shape[0]
        joint_attentions[0] = res_att_mat[0]
        for i in np.arange(1, layers):
            joint_attentions[i] = res_att_mat[i].dot(joint_attentions[i - 1])

        return joint_attentions


def handle_subwords(tokenwise_enc, first_token_ids, non_padded_ids, subwords):
    # merge subwords 
    if not subwords:
        # take the attention of only the first token of a word
        tokenwise_enc = tokenwise_enc[first_token_ids]
    else:
        # take the sum of the subwords's attention for a given word
        # id of the words's start
        attns = [np.sum(split_) for split_ in np.split(tokenwise_enc, first_token_ids, 0)[1:-1]]
        # the last have sum all the attention from the last non masked to the sep token (sep token is the last 1 in mask)
        last_sum = np.sum(tokenwise_enc[first_token_ids[-1] : non_padded_ids[-1]], 0)
        attns.append(last_sum)

        tokenwise_enc = np.array(attns)

    return tokenwise_enc


def mean_corrs(corr_dict):
    for l, feats in corr_dict.items():
        for f, corrs_list in feats.items():
            corr_dict[l][f] = np.mean(corrs_list)

    return corr_dict


def attention_correlations(d, model, subwords=False):
    LOGGER.info(f"Creating Attention Dataset, subwords sum : {subwords}...")

    corr_dict = dict()

    for i_layer in range(model.config.num_hidden_layers):
        corr_dict[i_layer] = dict() # for each layer i will put a dict that will contain an element for each feature
        for i_feature in range(model.num_labels):
            corr_dict[i_layer][i_feature] = [] # for each feature i will save a list of correlations over wich i will do a mean

    for input, target, mask in tqdm(list(zip(d.text_inputs, d.targets, d.masks))[:3]):
        with torch.no_grad():
            model_output = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]))

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

            sum_attention = handle_subwords(sum_attention, first_token_ids, non_padded_ids, subwords)

            # Taking the softmax does not make a difference for calculating correlation
            # It can be useful to scale the salience signal to the same range as the human attention
            relative_attention = scipy.special.softmax(sum_attention)

            word_targets = target[first_token_ids]

            # normalize features between [0,1]
            sum_features = np.sum(word_targets, axis=0)

            normalized_features = word_targets / sum_features

            # avoid division by 0 in the normalization of a full zero row
            normalized_features[np.isnan(normalized_features)] = 0

            for feat_i in range(normalized_features.shape[1]):
                sample_spearman_corr = spearmanr(relative_attention, normalized_features[:, feat_i])[0]
                # sometimes the spearmanr can return nan in case attn or targ has std = 0
                if not sample_spearman_corr is np.nan:
                    corr_dict[layer][feat_i].append(sample_spearman_corr)

    return mean_corrs(corr_dict)


def globenc_correlations(d, model, subwords=False):
    LOGGER.info(f"Creating GlobEnc Dataset, subwords sum : {subwords}...")
    
    corr_dict = dict() # an entry for each layer

    for i_layer in range(model.config.num_hidden_layers):
        corr_dict[i_layer] = dict() # for each layer i will put a dict that will contain an element for each feature
        for i_feature in range(model.num_labels):
            corr_dict[i_layer][i_feature] = [] # for each feature i will save a list of correlations over wich i will do a mean

    for input, target, mask in tqdm(list(zip(d.text_inputs, d.targets, d.masks))[:3]):
        # ---- GLOBENC
        with torch.no_grad():
            _, attentions, norms = model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]), output_norms=True, return_dict=False)

        num_layers = len(attentions)
        norm_nenc = torch.stack([norms[i][4] for i in range(num_layers)]).squeeze().cpu().numpy()

        # Aggregate and compute GlobEnc
        globenc = AttentionRollout().compute_flows([norm_nenc], output_hidden_states=True)[0]
        globenc = np.array(globenc)

        # ----

        for layer, enc in enumerate(globenc):
            non_padded_ids = np.where(mask == 1)[0]

            # We drop CLS and SEP tokens
            enc = enc[1:-1]

            # 2. For each word, we sum over the rollout to the other words to determine relative importance
            sum_enc = np.sum(enc, axis=0)

            sum_enc = sum_enc[non_padded_ids]

            first_token_ids = np.where(np.multiply.reduce(target[non_padded_ids] != -1, 1) > 0)[0]

            sum_enc = handle_subwords(sum_enc, first_token_ids, non_padded_ids, subwords)

            # Taking the softmax does not make a difference for calculating correlation
            # It can be useful to scale the salience signal to the same range as the human attention
            relative_enc = scipy.special.softmax(sum_enc)

            word_targets = target[first_token_ids]

            # normalize features between [0,1]
            sum_features = np.sum(word_targets, axis=0)

            normalized_features = word_targets / sum_features

            # avoid division by 0 in the normalization of a full zero row
            normalized_features[np.isnan(normalized_features)] = 0

            for feat_i in range(normalized_features.shape[1]):
                sample_spearman_corr = spearmanr(relative_enc, normalized_features[:, feat_i])[0]
                # sometimes the spearmanr can return nan in case attn or targ has std = 0
                if not sample_spearman_corr is np.nan:
                    corr_dict[layer][feat_i].append(sample_spearman_corr)

    return mean_corrs(corr_dict)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')

    args = parser.parse_args()
    config_file = args.config_file

    cf = Config.load_json(config_file)

    pretrained = cf.pretrained
    finetuned = cf.finetuned
    model_name = cf.model_name
    model_dir = cf.model_dir
    output_dir = cf.output_dir
    average = cf.average
    encode_attention = cf.encode_attention
    xlm = cf.xlm

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    # to use GlobEnc need to import models from modelling module.
    if xlm:
        tokenClassifier = XLMRobertaForTokenClassification
    else:
        tokenClassifier = BertForTokenClassification

    # Dataset
    d = GazeDataset(cf, tokenizer, cf.dataset)
    d.read_pipeline()
    d.randomize_data()

    if not finetuned: # downaload from huggingface
        print("download from hf")
        model = load_model_from_hf(tokenClassifier, model_name, pretrained, d.d_out)

    else: # load from disk
        print("load from disk")
        model = tokenClassifier.from_pretrained(model_dir, output_attentions=True, output_hidden_states=False)

    LOGGER.info(f"The loaded model has {model.config.num_hidden_layers} layers")

    if encode_attention:
        correlations = globenc_correlations(d, model, average)
        to_print = {"GlobEnc": correlations}
    else:
        correlations = attention_correlations(d, model, average)
        to_print = {"Attentions": correlations}

    with open(f"{output_dir}/corrs_results_{datetime.datetime.now()}.json", 'w') as f:
        json.dump(to_print, f)


if __name__ == "__main__":
    main()