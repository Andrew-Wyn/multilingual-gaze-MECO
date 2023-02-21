import os
import argparse

import torch

import pandas as pd

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # DataCollatorWithPadding,
    # EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    Trainer,
    TrainingArguments,
    # default_data_collator,
    set_seed,
)
from sklearn.model_selection import train_test_split


from gaze.utils import Config, LOGGER

def read_complexity_dataset(path=None):
    texts = list()
    labels = list()

    df = pd.read_csv(path)

    for _, row in df.iterrows():
        
        texts.append(row["SENTENCE"])

        label = 0

        for i in range(20):
            label += int(row[f"judgement{i+1}"])

        label = int(label/20)

        labels.append(label)

    return texts, labels


class ComplexityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_hf(model_name, pretrained, d_out=8):

    # Model
    LOGGER.info("Initiating model ...")
    if not pretrained:
        # initiate model with random weights
        LOGGER.info("Take randomized model")
        
        config = AutoConfig.from_pretrained(model_name, num_labels=d_out)
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        LOGGER.info("Take pretrained model")
    
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=d_out)

    return model


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

        print(outputs['logits'].shape)
        print(inputs['labels'])

        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'],
                                            inputs['labels'])
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":

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

    # set seed
    set_seed(cf.seed)

    # check if the output directory exists, if not create it!
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    texts, labels = read_complexity_dataset(args.dataset)

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.2, random_state=cf.seed)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2, random_state=cf.seed)


    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = ComplexityDataset(train_encodings, train_labels)
    val_dataset = ComplexityDataset(val_encodings, val_labels)
    test_dataset = ComplexityDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        num_train_epochs=cf.n_epochs,              # total number of training epochs
        per_device_train_batch_size=cf.train_bs,  # batch size per device during training
        per_device_eval_batch_size=cf.eval_bs,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=cf.weight_decay,               # strength of weight decay
    )

    model = load_model_from_hf(cf.model_name, not cf.random_weights, 7)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    train_result = trainer.train()

    trainer.save_model()

    # compute train results
    metrics = train_result.metrics

    # save train results
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # compute evaluation results
    metrics = trainer.evaluate()

    # save evaluation results
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)