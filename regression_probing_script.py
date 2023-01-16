import argparse
import os

import numpy as np

from gaze.dataset import GazeDataset
from gaze.utils import LOGGER, Config, load_model_from_hf
from gaze.prober import Prober
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


def main():
    parser = argparse.ArgumentParser(description='Regression Probing')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')

    # Load the script's arguments
    args = parser.parse_args()

    config_file = args.config_file

    # Load config file
    cf = Config.load_json(config_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, cf.dataset)
    d.read_pipeline()
    d.randomize_data()

    LOGGER.info("Read the data!!!")

    # Model
    if not cf.finetuned: # downaload from huggingface
        LOGGER.info("Model retrieving, download from hf...")
        model = load_model_from_hf(cf.model_name, cf.pretrained, d.d_out)
    else: # load from disk
        LOGGER.info("Model retrieving, load from disk...")
        model = AutoModelForTokenClassification.from_pretrained(cf.model_dir, output_attentions=False, output_hidden_states=True)
    
    LOGGER.info("Model retrieved!!!")

    prober = Prober(d, cf.feature_max, cf.output_dir)

    _ = prober.create_probing_dataset(model, mean=cf.average)
    prober.probe(cf.linear, cf.k_fold)


if __name__ == "__main__":
    main()