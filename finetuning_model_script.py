from gaze.dataset import GazeDataset
from gaze.utils import Config
from gaze.trainer import cross_validation
import torch
from sklearn.utils import shuffle
from gaze.utils import LOGGER, create_finetuning_optimizer, create_scheduler, randomize_model, Config, minMaxScaling, load_model_from_hf
from modeling.custom_bert import BertForTokenClassification
from gaze.dataloader import GazeDataLoader
from sklearn.preprocessing import MinMaxScaler
from gaze.trainer import GazeTrainer
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


import os
from torch.utils.tensorboard import SummaryWriter
import argparse


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')

    # Read the script's argumenents
    args = parser.parse_args()
    config_file = args.config_file

    # Load the .json configuration file
    cf = Config.load_json(config_file)

    # check if the output directory exists, if not create it!
    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)

    # Writer
    writer = SummaryWriter(args.output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, args.dataset)
    d.read_pipeline()
    d.randomize_data()

    # K-fold cross-validation
    
    train_losses, test_losses = cross_validation(cf, d, writer, DEVICE, k_folds=cf.k_folds)

    print("Train averaged losses:")
    print(train_losses)

    print("Test averaged losses:")
    print(test_losses)

    # Retrain over all dataset

    # min max scaler the targets
    d.targets = minMaxScaling(d.targets, feature_max=d.feature_max, pad_token=d.target_pad)

    # create the dataloader
    train_dl = GazeDataLoader(cf, d.text_inputs, d.targets, d.masks, d.target_pad, mode="train")

    # Model
    model = load_model_from_hf(cf.model_name, not cf.random_weights, d.d_out)

    # Optimizer
    optim = create_finetuning_optimizer(cf, model)

    # Scheduler
    scheduler = create_scheduler(cf, optim, train_dl)

    # Trainer
    trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, f"Final_Training",
                                DEVICE, writer=writer)
    trainer.train(save_model=True)

if __name__ == "__main__":
    main()