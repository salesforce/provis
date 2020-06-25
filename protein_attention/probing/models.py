"""Diagnostic classifiers for probing analysis. Based on TAPE models."""

import torch
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import PairwiseContactPredictionHead
from torch import nn


class ProteinBertForContactProbing(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


class ProteinBertForLinearSequenceToSequenceProbing(ProteinBertAbstractModel):
    """ProteinBert head for token-level prediction tasks (secondary structure, binding sites)"""

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)

        self.classify = LinearSequenceToSequenceClassificationHead(
            config.hidden_size,
            config.num_labels,
            ignore_index=-1,
            dropout=0.5)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        return outputs


class LinearSequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 ignore_index=-100,
                 dropout=0.1):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels))
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = (sequence_logits,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            classification_loss = loss_fct(
                sequence_logits.view(-1, self.num_labels), targets.view(-1))
            metrics = {
                'accuracy': accuracy(sequence_logits.view(-1, self.num_labels), targets.view(-1), self._ignore_index),
            }
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), sequence_logits


def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


def f1(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        unique_labels = set(labels * valid_mask)
        for label in unique_labels:
            if label not in (0, 1):
                raise NotImplementedError('Precision is only supported for binary labels')
        predictions = logits.float().argmax(-1)
        tp = (((predictions == 1) & (labels == 1)) * valid_mask).sum().float()
        fp = (((predictions == 1) & (labels == 0)) * valid_mask).sum().float()
        fn = (((predictions == 0) & (labels == 1)) * valid_mask).sum().float()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
