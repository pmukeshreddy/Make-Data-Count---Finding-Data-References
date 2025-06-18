class MultiTaskTransformer(PreTrainedModel):
    def __init__(self,config, num_pos_labels):
        super().__init__(config)
        self.num_primary_labels = config.num_labels
        self.num_pos_labels = num_pos_labels
        self.base_model = AutoModel.from_pretrained(config._name_or_path, config=config)
        self.dropout = nn.Dropout(0.1)
        self.primary_classifier = nn.Linear(config.hidden_size, self.num_primary_labels)
        self.pos_classifier = nn.Linear(config.hidden_size, self.num_pos_labels)
    def forward(self,input_ids=None, attention_mask=None, primary_label=None, pos_label=None):
        outputs = self.base_model(input_ids,attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:,0]
        cls_embedding = self.dropout(cls_embedding)

        primary_logits = self.primary_classifier(cls_embedding)
        pos_logits = self.pos_classifier(cls_embedding)

        return {"primary_logits": primary_logits, "pos_logits": pos_logits}
    
