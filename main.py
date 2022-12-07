from gaze.dataset import GazeDataset
from gaze.utils import Config
from gaze.trainer import cross_validation
import torch
from sklearn.utils import shuffle
from gaze.utils import LOGGER, create_finetuning_optimizer, create_scheduler, randomize_model, Config
from modeling.custom_bert import BertForTokenClassification
from gaze.dataloader import GazeDataLoader
from sklearn.preprocessing import MinMaxScaler
from gaze.trainer import GazeTrainer
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
    config_file = args.config_file

    cf = Config.load_json(config_file)

    eval_dir = "eval_dir"

    print(cf)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    writer = SummaryWriter(eval_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_pretrained, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, "datasets/it/all_mean_dataset.csv", "try")
    d.read_pipeline()

    # 10-fold cross-validation
    test_losses = cross_validation(cf, d, eval_dir, writer, DEVICE, k_folds=10)

    print(test_losses)

    # shuffle dataset with a seed, to take a reproducible sample
    shuffled_ids = shuffle(range(d.text_inputs.shape[0]), random_state=42)

    text_inputs = d.text_inputs[shuffled_ids]
    targets = d.targets[shuffled_ids]
    masks = d.masks[shuffled_ids]

    # retrain over all dataset
    # min max scaler the targets
    features = targets
    scaler = MinMaxScaler(feature_range=[0, d.feature_max])
    flat_features = [j for i in features for j in i]
    scaler.fit(flat_features)

    targets = [list(scaler.transform(i)) for i in features]

    # create the dataset
    train_d = list(zip(text_inputs, targets, masks))

    train_dl = GazeDataLoader(cf, train_d, d.target_pad, mode="train")

    # Model
    LOGGER.info("initiating model: ")
    model = BertForTokenClassification.from_pretrained(cf.model_pretrained, num_labels=d.d_out,
                                    output_attentions=False, output_hidden_states=False)

    if cf.random_weights is True:
        # initiate Bert with random weights
        print("randomizing weights")
        model = randomize_model(model)
        #print(model.classifier.weight.data)

    # optimizer
    optim = create_finetuning_optimizer(cf, model)

    # scheduler
    scheduler = create_scheduler(cf, optim, train_dl)

    # trainer
    trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, eval_dir, "task",
                                DEVICE, writer=writer)
    trainer.train()
    LOGGER.info(f"Training completed task")


if __name__ == "__main__":
    main()