import scipy
import torch
import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod

from scipy.stats import spearmanr

from gaze.utils import LOGGER


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


class Correlation(ABC):
    def __init__(self, d, model, subwords=False):
        self.d = d
        self.model = model
        self.subwords = subwords

    def _handle_subwords(self, tokenwise_enc, first_token_ids, non_padded_ids):
        # merge subwords 
        if not self.subwords:
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

    def _mean_corrs(self, corr_dict):
        for l, feats in corr_dict.items():
            for f, corrs_list in feats.items():
                corr_dict[l][f] = np.mean(corrs_list)

        return corr_dict

    @abstractmethod
    def compute_importance(self, input, mask):
        pass

    def compute_correlations(self):
        LOGGER.info(f"Creating GlobEnc Dataset, subwords sum : {self.subwords}...")
        
        corr_dict = dict() # an entry for each layer

        for i_layer in range(self.model.config.num_hidden_layers):
            corr_dict[i_layer] = dict() # for each layer i will put a dict that will contain an element for each feature
            for i_feature in range(self.model.num_labels):
                corr_dict[i_layer][i_feature] = [] # for each feature i will save a list of correlations over wich i will do a mean

        for input, target, mask in tqdm(list(zip(self.d.text_inputs, self.d.targets, self.d.masks))[:3]):
            
            importance_list = self.compute_importance(input, mask)

            for layer, importance in enumerate(importance_list):
                non_padded_ids = np.where(mask == 1)[0]

                importance = importance[non_padded_ids]

                first_token_ids = np.where(np.multiply.reduce(target[non_padded_ids] != -1, 1) > 0)[0]

                importance = self._handle_subwords(importance, first_token_ids, non_padded_ids)

                # Taking the softmax does not make a difference for calculating correlation
                # It can be useful to scale the salience signal to the same range as the human attention
                relative_importance = scipy.special.softmax(importance)

                word_targets = target[first_token_ids]

                # normalize features between [0,1]
                sum_features = np.sum(word_targets, axis=0)

                normalized_features = word_targets / sum_features

                # avoid division by 0 in the normalization of a full zero row
                normalized_features[np.isnan(normalized_features)] = 0

                for feat_i in range(normalized_features.shape[1]):
                    sample_spearman_corr = spearmanr(relative_importance, normalized_features[:, feat_i])[0]
                    # sometimes the spearmanr can return nan in case attn or targ has std = 0
                    if not sample_spearman_corr is np.nan:
                        corr_dict[layer][feat_i].append(sample_spearman_corr)

        return self._mean_corrs(corr_dict)


class AttentionCorrelation(Correlation):
    def __init__(self, d, model, subwords=False):
        super().__init__(d, model, subwords)

    def compute_importance(self, input, mask):
        importance_list = list()

        # ---- Attention
        with torch.no_grad():
            model_output = self.model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]))

        # once created the encoding, compress the encoding to have a vector for each sentence
        for attention_layer in model_output.attentions:
            # We use the first element of the batch because batch size is 1
            attention = attention_layer[0].numpy()

            # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
            # I also tried the sum once, but the result was even worse
            mean_attention = np.mean(attention, axis=0)

            # We drop padded tokens
            # mean_attention = mean_attention[non_padded_ids]

            # We drop CLS and SEP tokens
            mean_attention = mean_attention[1:-1]

            # 2. For each word, we sum over the attention to the other words to determine relative importance
            sum_attention = np.sum(mean_attention, axis=0)

            importance_list.append(sum_attention)

        # ----
        
        return importance_list


class GLOBENCCorrelation(Correlation):
    def __init__(self, d, model, subwords=False):
        super().__init__(d, model, subwords)

    def compute_importance(self, input, mask):
        # ---- GLOBENC
        with torch.no_grad():
            _, attentions, norms = self.model(input_ids=torch.as_tensor([input]), attention_mask=torch.as_tensor([mask]), output_norms=True, return_dict=False)

        num_layers = len(attentions)
        norm_nenc = torch.stack([norms[i][4] for i in range(num_layers)]).squeeze().cpu().numpy()

        # Aggregate and compute GlobEnc
        globenc = AttentionRollout().compute_flows([norm_nenc], output_hidden_states=True)[0]
        globenc = np.array(globenc)

        importance_list = list()

        # once created the encoding, compress the encoding to have a vector for each sentence
        for enc in globenc:
            # We drop CLS and SEP tokens
            enc = enc[1:-1]

            # 2. For each word, we sum over the rollout to the other words to determine relative importance
            sum_enc = np.sum(enc, axis=0)

            importance_list.append(sum_enc)

        # ----

        return importance_list
