from dataset import GazeDataset
from dataloader import GazeDataLoader
from trainer import GazeTrainer
from utils import LOGGER, create_finetuning_optimizer, create_scheduler, randomize_model
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
import os


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


def main():

    model = "prajjwal1/bert-mini"
    random_weights=False

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(10, tokenizer, "datasets/cluster_0_dataset.csv", "try")
    d.read_pipeline()

    train_dl = GazeDataLoader(256, 256, d.numpy["train"], d.target_pad, mode="train")
    val_dl = GazeDataLoader(256, 256, d.numpy["valid"], d.target_pad, mode="val") 

    # Model
    LOGGER.info("initiating model: ")
    model = AutoModelForTokenClassification.from_pretrained(model, num_labels=d.d_out,
                                    output_attentions=False, output_hidden_states=False)
    if random_weights is True:
        # initiate Bert with random weights
        print("randomizing weights")
        model = randomize_model(model)
        #print(model.classifier.weight.data)

    # optimizer
    optim = create_finetuning_optimizer(True, 0.001, 0.001, 0.001, model):

    # scheduler
    scheduler = create_scheduler(1, optim, train_dl)

    # trainer
    trainer = GazeTrainer(n_epocs, max_grad_norm, patience, model, train_dl, val_dl, optim, scheduler,
                                  "cpu", monitor="loss_all", monitor_mode="min")
    trainer.train()
    LOGGER.info(f"Training completed task")


if __name__ == "__main__":
    main()