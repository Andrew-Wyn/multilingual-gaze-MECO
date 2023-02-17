import math
from typing import Optional, Union, Tuple, Any, List, Dict, Mapping

import numpy as np
import torch
from attr import dataclass
from datasets import load_dataset
from torch.nn import MSELoss
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRObertaPretrainedModel, BertPreTrainedModel, XLMRobertaModel, BertModel, AutoTokenizer, training_args, TrainingArguments, Trainer, \
    get_scheduler
from transformers.data.data_collator import DataCollatorMixin, default_data_collator, InputDataClass
from transformers.utils import ModelOutput


class MultiTaskSequenceRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    multi_regression_output: Dict[int, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class XLMRobertaMultiTaskForSequenceRegression(XLMRObertaPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.xlm_roberta = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.regressors = nn.ModuleDict({
                    task: nn.Linear(config.hidden_size, 1) for
                                        task in range(self.num_labels)
                                    })
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]

        pooled_output = self.dropout(pooled_output)

        multi_regression_output = {}
        loss = None
        loss_fct = MSELoss()

        for task in range(self.num_labels):
                output_task = self.regressors[task](pooled_output)
                multi_regression_output[task] = output_task
                
                if labels is not None:
                    loss_ = loss_fct(output_task, labels)
                    if loss is None:
                        loss = loss_
                    else:
                        loss += loss_

        if not return_dict:
            output = (multi_regression_output,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiTaskSequenceRegressionOutput(
            loss=loss,
            multi_regression_output=multi_regression_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics(self, eval_pred):
    print(eval_pred)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return self.metric.compute(predictions=predictions, references=labels)