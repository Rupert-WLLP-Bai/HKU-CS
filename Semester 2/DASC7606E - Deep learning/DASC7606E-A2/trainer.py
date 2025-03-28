import torch
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics

class BiLSTMTrainer(Trainer):
    """自定义训练器，处理XLM-RoBERTa+BiLSTM模型输出"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """重写计算损失函数方法，支持num_items_in_batch参数"""
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """重写预测步骤方法"""
        has_labels = all(inputs.get(k) is not None for k in ["labels"])
        inputs = self._prepare_inputs(inputs)
        
        # 执行前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                # 不要调用 .item()，保持为张量
                loss = outputs["loss"] if "loss" in outputs else None
            else:
                logits = outputs[1]
                loss = outputs[0]  # 去掉 .item()
        
        if prediction_loss_only:
            return (loss, None, None)
        
        if has_labels:
            labels = inputs["labels"]
        else:
            labels = None
            
        return (loss, logits, labels)


def create_training_arguments() -> TrainingArguments:
    """创建训练参数"""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        
        # 评估与保存策略
        eval_strategy="steps",  # 使用新参数名
        eval_steps=500,
        save_strategy="steps",
        save_steps=2000,
        logging_steps=500,
        save_total_limit=5,
        
        # 学习率与优化器设置 - 调整以适应BiLSTM
        learning_rate=8e-5,
        # warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # 批量大小与累积
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        
        # 正则化与稳定性
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        
        # 混合精度
        fp16=True,
        
        # 模型选择
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    
    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> BiLSTMTrainer:
    """构建训练器对象"""
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = create_training_arguments()
    
    # 使用自定义训练器，修正参数名
    trainer = BiLSTMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,  # 推荐使用 processing_class 而非 tokenizer
    )
    
    return trainer