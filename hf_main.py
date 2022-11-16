import os
import torch
import numpy as np

from torch.nn import BCEWithLogitsLoss
from gaze.dataset import GazeDataset
from datasets import Dataset


CACHE_DIR = f"{os.getcwd()}/.hf_cache/"


def dataset_transform(input_ids, targets):
    # transform list of lists in a list of datasets
    return_list_dicts = list()

    for txt, labels in zip(input_ids, targets):
        return_list_dicts.append({
            "tokens": txt,
            "labels": labels
        })

    return return_list_dicts

if __name__ == "__main__":
    d = GazeDataset(None, None, "datasets/cluster_0_dataset.csv", None)

    d.load_data()

    text_inputs = d.text_inputs
    targets = d.targets

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=CACHE_DIR)

    train_dataset_list = dataset_transform(text_inputs["train"], targets["train"])
    train_dataset = Dataset.from_list(train_dataset_list)

    test_dataset_list = dataset_transform(text_inputs["test"], targets["test"])
    test_dataset = Dataset.from_list(test_dataset_list)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append([-100 for _ in range(8)])
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append([-100 for _ in range(8)])
                previous_word_idx = word_idx


            # pad by my self, since the fact, padding in datacollator doesn't works
            if len(label_ids) < 126:
                for _ in range(126 - len(label_ids)):
                    label_ids.append([-100 for _ in range(8)])

            labels.append(label_ids)        

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

    from transformers import DataCollatorForTokenClassification

    # the datacollator will pad with one-dimension elements

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=8)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    class MultiLabelTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            if class_weights is not None:
                class_weights = class_weights.to(self.args.device)
                # logging.info(f"Using multi-label classification with class weights", class_weights)
            self.loss_fct = BCEWithLogitsLoss(weight=class_weights)

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            labels = inputs.pop("labels")
            outputs = model(**inputs)

            print(labels.shape)
            print(outputs.logits.shape)

            try:
                loss = self.loss_fct(outputs.logits.view(-1, model.num_labels), labels.view(-1, ))
            except AttributeError:  # DataParallel
                loss = self.loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss
    

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

