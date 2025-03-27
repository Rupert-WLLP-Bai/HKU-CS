from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,               # 增加轮次
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        
        # 评估与保存策略
        eval_strategy="steps",
        eval_steps=500,                   # 更频繁评估
        save_strategy="steps",
        save_steps=2000,                   # 更频繁保存
        logging_steps=500,                # 更频繁记录
        save_total_limit=3,               # 节省空间
        
        # 学习率与优化器设置
        learning_rate=5e-5,               # 降低学习率
        warmup_ratio=0.1,                 # 添加预热阶段
        lr_scheduler_type="linear",       # 改为线性衰减
        
        # 批量大小与累积
        per_device_train_batch_size=8,    # 降低单批量大小
        per_device_eval_batch_size=16,    # 评估批量保持不变
        gradient_accumulation_steps=4,    # 增加累积步骤
        
        # 正则化与稳定性
        weight_decay=0.01,                # 调整权重衰减
        max_grad_norm=2.0,                # 增加梯度裁剪阈值
        label_smoothing_factor=0.1,       # 添加标签平滑
        
        # 混合精度
        fp16=True,                        # 保持混合精度训练
        
        # 模型选择与早停
        load_best_model_at_end=True,      # 加载最佳模型
        metric_for_best_model="f1",       # 以F1为标准
        greater_is_better=True,           # 更高的F1更好
    )

    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for token classification.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.
    """
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args: TrainingArguments = create_training_arguments()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
