class MultiTaskFocalLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, alpha=0.25, gamma=2.0, multitask_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.multitask_loss_weight = multitask_loss_weight
    def compute_loss(self,model,inputs,returns_output=False):
        primary_labels = inputs.pop("primary_label")
        pos_labels = inputs.pop("pos_label")

        outputs = model(**inputs)
        primary_logits = outputs["primary_logits"]
        pos_logits = outputs["pos_logits"]

        #primary task loss
        ce_loss = CrossEntropyLoss(weight=self.class_weights.to(self.device),reduction="none")(primary_logits,primary_labels)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()
        #secondary loss
        pos_loss_fct = CrossEntropyLoss()
        pos_loss = pos_loss_fct(pos_logits,pos_labels)

        # total loss
        total_loss = (1 - self.multitask_loss_weight) * focal_loss + self.multitask_loss_weight * pos_loss
        return (total_loss, outputs) if return_outputs else total_loss
