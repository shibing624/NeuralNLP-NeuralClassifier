#!/usr/bin/env python
# coding: utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    BertConfig,
    AutoModel,
)
from model.classification.classifier import Classifier

from dataset.classification_dataset import ClassificationDataset as cDataset


class BertHMCN(Classifier):
    """ Implement HMCN(Hierarchical Multi-Label Classification Networks)
        Reference: "Hierarchical Multi-Label Classification Networks"
    """

    def __init__(self, dataset, config):
        super(BertHMCN, self).__init__(dataset, config)
        self.hierarchical_depth = config.HMCN.hierarchical_depth
        self.hierarchical_class = dataset.hierarchy_classes
        self.global2local = config.HMCN.global2local
        self.bert = AutoModel.from_pretrained(config.bert_model)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()
        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.bert_hidden_size + self.hierarchical_depth[i - 1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=config.train.hidden_layer_dropout)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i - 1])
                ))

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], len(dataset.label_map))
        self.linear.apply(self._init_weight)
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)
        self.classifier = nn.Linear(self.bert_hidden_size, len(dataset.label_map))

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)

    def get_parameter_optimizer_dict(self):
        params = super(BertHMCN, self).get_parameter_optimizer_dict()
        params.append({'params': self.bert.parameters()})
        params.append({'params': self.local_layers.parameters()})
        params.append({'params': self.global_layers.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """ Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        # bert input: input_ids, attention_mask, token_type_ids, position_ids, head_mask
        outputs = self.bert(
            batch[cDataset.DOC_INPUT_IDS].to(self.config.device),
            attention_mask=batch[cDataset.DOC_ATTENTION_MASK].to(self.config.device),
            token_type_ids=batch[cDataset.DOC_TOKEN_TYPE_IDS].to(self.config.device),
        )
        doc_embedding = outputs[1]
        local_layer_outputs = []
        global_layer_activation = doc_embedding
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers) - 1:
                global_layer_activation = torch.cat((local_layer_activation, doc_embedding), 1)
            else:
                global_layer_activation = local_layer_activation

        global_layer_output = self.linear(global_layer_activation)
        local_layer_output = torch.cat(local_layer_outputs, 1)
        last_loss = 0.5 * global_layer_output + 0.5 * local_layer_output

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        last_loss = logits
        return global_layer_output, local_layer_output, last_loss


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
