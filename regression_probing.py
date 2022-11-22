import argparse
import os
from collections import defaultdict
from tqdm import tqdm


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


def create_probing_dataset(cf, tokenizer, model, mean=False):
    # one entry per layer each of them is a 
    probing_dataset = dict()

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/cluster_0_dataset.csv", "try")
    d.read_pipeline()

    # train_dl = GazeDataLoader(cf, d.numpy["train"], d.target_pad, mode="train")
    # val_dl = GazeDataLoader(cf, d.numpy["valid"], d.target_pad, mode="val") 

    print(model.config.num_hidden_layers)

    probing_dataset["train"] = defaultdict(list)

    for input, target, mask in tqdm(d.numpy["test"]):
        # print(input)
        # print(target.shape)
        # print(mask.shape)

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
                input.append(hidden_state[0, np.where(non_masked_els)[0][-1], :])
                input = np.array(input)


            output = target[non_masked_els, :]

            probing_dataset["train"][layer].append((input, output))
        
    # concatenate the inputs and outputs
    for layer in range(model.config.num_hidden_layers):
        input_list = []
        output_list = []
        
        for input, output in probing_dataset["train"][layer]:
            input_list.append(input)
            output_list.append(output)

        input_list = np.array(input_list)
        output_list = np.array(output_list)

        probing_dataset["train"][layer] = (input_list, output_list)
        

    print(probing_dataset["train"][1][0].shape)
    

            

        


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


def main():
    parser = argparse.ArgumentParser(description='Regression Probing')
    parser.add_argument('-m' ,'--model', dest='model_name', action='store',
                        help=f' of the model to retrieve from the HF repository')
    parser.add_argument('-d' ,'--model_dir', dest='model_dir', action='store',
                        help=f'Relative path of the pretrained model')
    parser.add_argument('-p' ,'--pretrained', dest='pretrained', action=argparse.BooleanOptionalAction,
                        help=f'If needed a pretrained model')
    parser.add_argument('-f' ,'--finetuned', dest='finetuned', action=argparse.BooleanOptionalAction,
                        help=f'If needed a finetuned model')
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

    pretrained = args.pretrained
    finetuned = args.finetuned
    model_name = args.model_name
    model_dir = args.model_dir
    config_file = args.config_file

    cf = Config.load_json(config_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if not finetuned: # downaload from huggingface
        print("download from hf")

        model = load_model_from_hf(model_name, pretrained)

    else: # load from disk
        print("load from disk")
        model = AutoModelForTokenClassification.from_pretrained(model_dir, output_attentions=False, output_hidden_states=True)

    
    _ = create_probing_dataset(cf, tokenizer, model)

if __name__ == "__main__":
    main()