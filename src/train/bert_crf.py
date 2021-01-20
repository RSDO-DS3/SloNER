import torch
import pandas as pd

from transformers import BertPreTrainedModel, BertModel
from torch.nn.functional import log_softmax
from TorchCRF import CRF


class BertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels)

    def forward(self, input_ids, attn_masks, labels=None):
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            # loss = -self.crf.forward(log_softmax(emission, 2), labels, mask=attn_masks)
            loss = -self.crf.forward(emission, labels, mask=attn_masks)
            return loss
        else:
            pred = self.crf.viterbi_decode(emission, attn_masks)
            return pred


if __name__ == '__main__':
    bert_crf = BertCRFForTokenClassification.from_pretrained('./data/models/cro-slo-eng-bert', num_labels=9)
