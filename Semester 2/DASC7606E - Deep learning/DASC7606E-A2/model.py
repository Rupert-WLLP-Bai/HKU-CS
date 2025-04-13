import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
from constants import MODEL_CHECKPOINT, LABEL_TO_ID

class BertBiLSTMCRF(nn.Module):
    """在BERT基础上添加BiLSTM层和CRF层的NER模型"""
    def __init__(self, model_checkpoint, num_labels, hidden_dropout_prob=0.1, lstm_hidden_size=768, lstm_layers=2):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)  # 添加CRF层
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)
        
        # 应用BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size)
        
        # 应用分类器
        logits = self.classifier(lstm_output)  # (batch_size, seq_length, num_labels)
        
        # 计算CRF损失
        loss = None
        if labels is not None:
            # CRF的损失计算
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": None,
            "attentions": None
        }


class BertLstm(nn.Module):
    """在BERT基础上添加BiLSTM层的NER模型"""
    def __init__(self, model_checkpoint, num_labels, hidden_dropout_prob=0.1, lstm_hidden_size=768, lstm_layers=2):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # print("Labels:", labels)
        # print("Labels shape:", labels.shape)
        # print("Unique label values:", torch.unique(labels))
        # print("Attention mask:", attention_mask)
        # print("Attention mask shape:", attention_mask.shape)
        
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)
        
        # 应用BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_length, lstm_hidden_size)
        
        # 应用分类器
        logits = self.classifier(lstm_output)  # (batch_size, seq_length, num_labels)
        
        # Preprocess labels to replace -100 with 0 (or any valid label index)
        if labels is not None:
            labels = labels.clone()  # Avoid modifying the original labels tensor
            labels[labels == -100] = 0  # Replace -100 with 0 (valid label index)
            
        # print("Logits shape:", logits.shape)
        
        # compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            
            
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": None,
            "attentions": None
        }
        

class Bert(nn.Module):
    """BERT模型"""
    def __init__(self, model_checkpoint, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        
        logits = self.classifier(sequence_output)  # (batch_size, seq_length, num_labels)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": None,
            "attentions": None
        }

def initialize_model():
    model = BertLstm(
        model_checkpoint=MODEL_CHECKPOINT,
        num_labels=len(LABEL_TO_ID),
        hidden_dropout_prob=0.1,
        lstm_hidden_size=768,
        lstm_layers=2
    )
    
    # model = Bert(
    #     model_checkpoint=MODEL_CHECKPOINT,
    #     num_labels=len(LABEL_TO_ID)
    # )
    
    return model