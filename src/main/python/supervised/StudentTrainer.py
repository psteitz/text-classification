from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self,args,train_dataset,eval_dataset,tokenizer,data_collator,compute_metrics, teacher_model):
        super().__init__(args, train_dataset, eval_dataset,tokenizer, data_collator, compute_metrics)
        self.teacher_model = teacher_model
                 
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(device=model.device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss