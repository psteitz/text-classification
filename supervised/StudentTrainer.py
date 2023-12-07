
from torch import nn, torch
from transformers import Trainer
from inference import Inference

YAHOO_CLASSES = [
    "society or culture",
    "science or mathematics",
    "health",
    "education or reference",
    "computers or internet",
    "sports",
    "business or finance",
    "entertainment or music",
    "family or relationships",
    "politics or government"
]


class StudentTrainer(Trainer):
    """
    StudentTrainer: extends huggingface Trainer class.

    Overrides compute_loss method to replace the one-hot target distribution  with the teacher's
    predicted distribution if teacher gets it right and puts enough mass on the correct label.

    """
    count = 0
    Teacher_correct_count = 0

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer,
                 data_collator, compute_metrics, teacher_model_dir, threshhold):
        super().__init__(model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, None, compute_metrics)
        self.teacher_model_dir = teacher_model_dir
        self.threshold = threshhold

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # print("labels", labels)
        batch_size = len(labels)
        # labels is a one-dimensional tensor of length = batch size
        # labels[0] is a list of correct labels for examples in the batch
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # logits is a two-dimensional tensor of shape (batch size, number of classes)
        # Each row is a vector of logits for one example in the batch
        if StudentTrainer.count % 100 == 0:
            print("batch count", StudentTrainer.count)
            # print("model-in-training logits",logits)
            if StudentTrainer.Teacher_correct_count > 0:
                print("Teacher correct rate", StudentTrainer.Teacher_correct_count /
                      (StudentTrainer.count * batch_size))

        # Run the teacher model on inputs.
        # For examples where the teacher does well enough (gets it right and puts enough mass on the correct label),
        # use the teacher's predicted distribution as the target distribution; ptherwise use one-hot.
        inference = Inference(model_directory=self.teacher_model_dir)
        teacher_outputs = inference.model(**inputs)
        teacher_logits = teacher_outputs.get("logits")
        # print("teacher logits", teacher_logits)
        # Initialize targets to tensor of one-hot vectors converted to floats so each is a distribution
        targets = torch.nn.functional.one_hot(labels, num_classes=10).float()
        # print("targets", targets)

        # Examine the teacher_logits tensor, checking to see if the teacher response is correct and
        # stable.  Stable means that the difference between the softmax score of the correct label
        # is at least self.threshold greater than the softmax score of the next highest label.
        # If so, replace the corresponding one-hot target with the teacher's predicted distribution
        for i in range(batch_size):
            correct_label = labels[i]
            teacher_predicted_distribution = torch.softmax(
                teacher_logits[i], dim=0)
            if torch.argmax(teacher_predicted_distribution) == correct_label:
                # teacher got it right
                # Now check to make sure the difference between the softmax score of the correct label
                # is at least self.threshold greater than the softmax score of the next highest label
                sorted_distribution = torch.sort(
                    teacher_predicted_distribution, descending=True)
                if sorted_distribution[0][0] - sorted_distribution[0][1] > self.threshold:
                    # Replace the one-hot target with the teacher's predicted distribution
                    targets[i, :] = teacher_predicted_distribution
                    StudentTrainer.Teacher_correct_count += 1

        # set loss function to cross entropy loss between logits and targets
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, targets)
        StudentTrainer.count += 1
        return (loss, outputs) if return_outputs else loss
