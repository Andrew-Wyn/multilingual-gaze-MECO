from gaze.dataset import GazeDataset
from gaze.utils import Config
from gaze.trainer import cross_validation
import torch
from sklearn.utils import shuffle
from gaze.utils import LOGGER, create_finetuning_optimizer, create_scheduler, randomize_model, Config, minMaxScaling
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

    args = parser.parse_args()
    config_file = args.config_file

    cf = Config.load_json(config_file)

    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)

    writer = SummaryWriter(cf.output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, cf.dataset, "try")
    d.read_pipeline()
    d.randomize_data()

    # 10-fold cross-validation
    train_losses, test_losses = cross_validation(cf, d, writer, DEVICE, k_folds=10)

    print("Train losses:")
    print(train_losses)

    print("Test Losses:")
    print(test_losses)

    # retrain over all dataset

    # min max scaler the targets
    d.targets = minMaxScaling(d.targets, feature_max=d.feature_max)

    # create the dataloader
    train_dl = GazeDataLoader(cf, d.text_inputs, d.targets, d.masks, d.target_pad, mode="train")

    # Model
    LOGGER.info("initiating model:")
    if cf.random_weights:
        # initiate model with random weights
        LOGGER.info("Take randomized model:")
        config = AutoConfig.from_pretrained(cf.model_name, num_labels=d.d_out,
                                    output_attentions=False, output_hidden_states=False)
        model = AutoModelForTokenClassification.from_config(config)
    else:
        LOGGER.info("Take pretrained model:")
        model = BertForTokenClassification.from_pretrained(cf.model_name, num_labels=d.d_out,
                            output_attentions=False, output_hidden_states=False)

    # optimizer
    optim = create_finetuning_optimizer(cf, model)

    # scheduler
    scheduler = create_scheduler(cf, optim, train_dl)

    # trainer
    trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, f"Final_Training",
                                DEVICE, writer=writer)
    trainer.train(save_model=True)

if __name__ == "__main__":
    main()