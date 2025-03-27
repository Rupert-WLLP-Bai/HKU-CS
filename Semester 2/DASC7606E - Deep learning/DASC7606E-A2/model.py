import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from constants import LABEL_TO_ID, MODEL_CHECKPOINT

class XLMRobertaBiLSTMForNER(nn.Module):
    """在XLM-RoBERTa基础上添加BiLSTM层的NER模型"""
    def __init__(self, model_checkpoint, num_labels, hidden_dropout_prob=0.1, lstm_hidden_size=768, lstm_layers=2):
        super().__init__()
        self.num_labels = num_labels
        
        # 加载XLM-RoBERTa编码器
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.transformer = AutoModel.from_pretrained(model_checkpoint)
        
        # 获取Transformer输出维度
        self.hidden_size = self.config.hidden_size
        
        # 添加Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # 添加双向LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size // 2,  # 双向各一半
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # 添加分类层
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        
        # 初始化权重
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """前向传播"""
        # XLM-RoBERTa不使用token_type_ids
        outputs = self.transformer(
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
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # 在序列维度上展平
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
    """
    初始化用于命名实体识别的模型。
    """
    num_labels = len(LABEL_TO_ID)
    
    # 使用自定义的XLM-RoBERTa+BiLSTM模型
    model = XLMRobertaBiLSTMForNER(
        model_checkpoint=MODEL_CHECKPOINT,  # 使用constants.py中定义的MODEL_CHECKPOINT
        num_labels=num_labels,
        hidden_dropout_prob=0.2,
        lstm_hidden_size=768,
        lstm_layers=2
    )
    
    return model