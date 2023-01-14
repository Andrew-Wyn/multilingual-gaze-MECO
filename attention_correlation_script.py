import argparse
import datetime
import json
import os

from gaze.dataset import GazeDataset
from gaze.utils import LOGGER, randomize_model, Config
from gaze.correlation import AttentionCorrelation, GLOBENCCorrelation

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


# different definition with respect to the one in utils.py, this implementation has
# another parameter (tokenClassifier) needed since it will load a custom version of NLM.
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
        corr = GLOBENCCorrelation(d, model, average)
        correlations = corr.compute_correlations()
        to_print = {"GlobEnc": correlations}
    else:
        corr = AttentionCorrelation(d, model, average)
        correlations = corr.compute_correlations()
        to_print = {"Attentions": correlations}

    with open(f"{output_dir}/corrs_results_{datetime.datetime.now()}.json", 'w') as f:
        json.dump(to_print, f)


if __name__ == "__main__":
    main()