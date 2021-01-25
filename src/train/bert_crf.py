import torch
import pandas as pd
import numpy as np

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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        print(f"OUTPUTS {np.ndim(outputs)}")
        print(f"OUTPUTS[0] {outputs[0].shape}")
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attention_mask = attention_mask.type(torch.uint8)
        if self.training:
            # loss = -self.crf.forward(log_softmax(emission, 2), labels, mask=attention_mask)
            loss = -self.crf.forward(emission, labels, mask=attention_mask)
            print(f"CRF LOSS DIMENSION: {loss.shape}, content: {loss}")
            return loss
        else:
            pred = np.array(self.crf.viterbi_decode(emission, attention_mask))
            print(f"CRF PRED DIMENSION: {pred.shape}, content: {pred}")
            return torch.FloatTensor(pred).to("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    bert_crf = BertCRFForTokenClassification.from_pretrained('./data/models/cro-slo-eng-bert', num_labels=9)
