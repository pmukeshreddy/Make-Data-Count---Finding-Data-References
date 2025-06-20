import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset , ClassLabel
# CRITICAL: Import the new base model class
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import re
import spacy

def advance_finetune(training_data,num_pos_labels):
    df = pd.DataFrame(training_data)
    
    # We revert to using the standard 'labels' name, as the new base model expects it
    if "primary_label" in df.columns:
        df = df.rename(columns={"primary_label": "labels"})

    class_weights = compute_class_weight("balanced",classes=np.array([0,1,2]),y=df["labels"])
    class_weight_tensors = torch.tensor(class_weights,dtype=torch.float)

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # The config must now be set to output hidden states
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=3, output_hidden_states=True)

    # CRITICAL CHANGE: We now load the ForSequenceClassification version of the model
    base_transformer_model = AutoModelForSequenceClassification.from_pretrained(config._name_or_path, config=config)
    
    dataset = Dataset.from_pandas(df)
    def tokenise_function(examples):
        return tokenizer(examples["text"],padding="max_length",truncation=True,max_length=128)
    tokenized_dataset = dataset.map(tokenise_function, batched=True)
    
    tokenized_dataset = tokenized_dataset.cast_column("labels", ClassLabel(num_classes=config.num_labels))
    tokenized_dataset = tokenized_dataset.select_columns(['input_ids', 'attention_mask', 'labels', 'pos_label'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'pos_label'])
    
    train_val_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    train_dataset = train_val_split['train']
    eval_dataset = train_val_split['test']
    
    # The data collator is now simpler as no key renaming is needed
    def collect_fn(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch]),
            'pos_label': torch.stack([x['pos_label'] for x in batch]),
        }

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_dropout=0.1,lora_alpha=16,bias="none",target_modules=['query_proj', 'value_proj'])
    peft_model = get_peft_model(base_transformer_model,lora_config)
    peft_model.print_trainable_parameters()

    model = MultiTaskTransformer(config, num_pos_labels=num_pos_labels, base_model_instance=peft_model)

    training_args = TrainingArguments(
        output_dir="./deberta_advanced_finetuned",
        learning_rate=2e-4, 
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16, 
        num_train_epochs=1,
        weight_decay=0.01, 
        logging_strategy="epoch",
        eval_strategy="epoch",    
        save_strategy="epoch", 
        load_best_model_at_end=True,
        logging_dir="./logs",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = MultiTaskFocalLossTrainer(model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collect_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weight_tensors)

    print("DEBUG: Before trainer.train() call.")
    trainer.train()
    print("--- Trainer.train() completed ---")
