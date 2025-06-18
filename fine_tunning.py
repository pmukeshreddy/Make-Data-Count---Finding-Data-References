def advance_finetune(training_data,num_pos_labels):
    df = pd.DataFrame(training_data)

    class_weights = compute_class_weight("balanced",classes=np.unique(df["primary_label"]),y=df["primary_label"])
    class_weight_tensors = torch.tensor(class_weights,dtype=torch.float)

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    config = AutoConfig.from_pretrained(MODEL_NAME,num_labels=3)
    model = MultiTaskTransformer(config,num_pos_labels=num_pos_labels)

    dataset = Dataset.from_pandas(df)
    def tokenise_function(examples):
        return tokenizer(examples["text"],padding="max_length",truncation=True,max_length=128)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("primary_label", "label")
    train_val_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    train_dataset = train_val_split['train']
    eveal_dataset = train_val_split['test']
    train_dataset = train_dataset.rename_column("label", "primary_label")
    eval_dataset = eval_dataset.rename_column("label", "primary_label")
    def collect_fn(batch):
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch]),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
            'primary_label': torch.tensor([item['primary_label'] for item in batch]),
            'pos_label': torch.tensor([item['pos_label'] for item in batch])
        }
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_dropout=0.1,lora_alpha=16,bias=None,target_modules=['query_proj', 'value_proj'])
    peft_model = get_peft_model(model,lora_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(output_dir="./deberta_advanced_finetuned",learning_rate=2e-4, 
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16, 
        num_train_epochs=3, 
        weight_decay=0.01, 
        evaluation_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True)
    trainer = MultiTaskFocalLossTrainer(model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor)
    trainer.train()
