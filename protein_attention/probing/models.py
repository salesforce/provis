"""Diagnostic classifiers for probing analysis. Based on TAPE models."""

import torch
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import PairwiseContactPredictionHead
from torch import nn
from transformers import BertModel
import torch.nn.functional as F


class ProteinBertForLinearSequenceToSequenceProbingFromAttention(ProteinBertAbstractModel):
    """Bert head for token-level prediction tasks (secondary structure, binding sites) from attention weights"""

    def __init__(self, config):
        super().__init__(config)
        config.output_attentions = True
        self.bert = ProteinBertModel(config)
        self.predict = LinearSequenceToSequenceClassificationFromAttentionHead(config,
                                                                  ignore_index=-1)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)
        attention = outputs[-1]
        print('Sum of attentoins', attention[0][0, 0, 0].sum())
        last_layer_attention = attention[-1]
        outputs = self.predict(last_layer_attention, targets) #+ outputs[2:]
        return outputs

class LinearSequenceToSequenceClassificationFromAttentionHead(nn.Module):

    def __init__(self,
                 config,
                 ignore_index=-100,
                 dropout=0.1,
                 num_top_weights=10):
        super().__init__()
        if hasattr(config, 'probing_heads'):
            self.probing_heads = config.probing_heads
        else:
            self.probing_heads = list(range(config.num_attention_heads))
        self.classify = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len(self.probing_heads) * num_top_weights, config.num_labels))
        self.num_labels = config.num_labels
        self._ignore_index = ignore_index
        self.num_top_weights = num_top_weights

    def forward(self, attention, targets=None):
        """ Args:
            attention: tensor of shape
                    ``(batch_size, num_heads, sequence_length, sequence_length)``
        """
        batch_size = attention.shape[0]
        seq_len = attention.shape[2]
        assert attention.shape[3] == seq_len
        head_attentions = []
        for head in self.probing_heads:
            head_attention = attention[:, head].squeeze(1)
            assert head_attention.shape == (batch_size, seq_len, seq_len)
            head_attentions.append(head_attention)

        stacked = torch.stack(head_attentions)
        assert stacked.shape == (len(self.probing_heads), batch_size, seq_len, seq_len)
        stacked = stacked.permute(1, 0, 3, 2)
        assert stacked.shape == (batch_size, len(self.probing_heads), seq_len, seq_len)
        # Now dim 2 has the attention TO a particular position and dim 3 has the attention FROM a position

        features = stacked.topk(self.num_top_weights)[0]
        assert features.shape == (batch_size, len(self.probing_heads), seq_len, self.num_top_weights)
        # Last dimension is the top K attention weights to a given sequence position

        features = features.permute(0, 2, 1, 3)
        assert features.shape == (batch_size, seq_len, len(self.probing_heads),  self.num_top_weights)

        features = features.flatten(start_dim=2)
        # Flatten last two dimension to create a single feature vector for each sequence position
        assert features.shape == (batch_size, seq_len, len(self.probing_heads) * self.num_top_weights)

        sequence_logits = self.classify(features)
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



class ProteinBertForContactPredictionFromAttention(ProteinBertAbstractModel):
    """Bert head for token-pair contact prediction from attention weights"""

    def __init__(self, config):
        super().__init__(config)
        config.output_attentions = True
        self.bert = ProteinBertModel(config)
        self.predict = PairwiseContactPredictionFromAttentionHead(config,
                                                                  ignore_index=-1)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)
        attention = outputs[-1]
        last_layer_attention = attention[-1]
        outputs = self.predict(last_layer_attention, protein_length, targets) #+ outputs[2:]
        return outputs


class PairwiseContactPredictionFromAttentionHead(nn.Module):

    def __init__(self, config, ignore_index=-100):
        super().__init__()
        if hasattr(config, 'probing_heads'):
            self.probing_heads = config.probing_heads
        else:
            self.probing_heads = list(range(config.num_attention_heads))
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(len(self.probing_heads), 2))
        self._ignore_index = ignore_index

    def forward(self, attention, sequence_lengths, targets=None):
        """ Args:
            attention: tensor of shape
                    ``(batch_size, num_heads, sequence_length, sequence_length)``
        """

        batch_size = attention.shape[0]
        seq_len = attention.shape[2]
        assert attention.shape[3] == seq_len
        head_attentions = []
        for head in self.probing_heads:
            head_attention = attention[:, head].squeeze(1)
            assert head_attention.shape == (batch_size, seq_len, seq_len)
            head_attentions.append(head_attention)

        num_features = len(head_attentions)
        stacked = torch.stack(head_attentions)
        assert stacked.shape == (num_features, batch_size, seq_len, seq_len)
        pairwise_features = stacked.permute(1, 2, 3, 0)
        assert pairwise_features.shape == (batch_size, seq_len, seq_len, num_features)
        pairwise_features = (pairwise_features + pairwise_features.transpose(1,2))/2 # Mean attention from both directions
        prediction = self.classifier(pairwise_features)
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove start/stop tokens
        assert prediction.shape == (batch_size, seq_len - 2, seq_len - 2, 2)
        outputs = (prediction,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            loss_prediction = prediction.view(-1, 2)
            assert loss_prediction.shape == (batch_size * (seq_len - 2)**2, 2)
            loss_targets = targets.view(-1)
            assert loss_targets.shape == (batch_size * (seq_len - 2)**2, )
            contact_loss = loss_fct(loss_prediction, loss_targets)
            metrics = {'precision_at_l5':
                       self.compute_precision_at_l5(sequence_lengths, prediction, targets)}
            loss_and_metrics = (contact_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total



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

class BertForLinearSequenceToSequenceProbing(ProteinBertAbstractModel):
    """Bert head for token-level prediction tasks (secondary structure, binding sites)"""

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

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
