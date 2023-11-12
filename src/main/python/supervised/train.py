#!/usr/bin/env python

"""
train.py: trains huggingface distilbert-base-uncaset model to classify texts in
hf yahoo_answers_topics dataset
"""

import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer, AutoTokenizer, pipeline
from datasets import load_dataset

# Huggingace model name for distilbert base uncased
bert_model = "distilbert-base-uncased"

# load HuggingFace dataset
yahoo_answers_topics = load_dataset("yahoo_answers_topics")

# use AutoTokenizer for BERT_MODEL
tokenizer = AutoTokenizer.from_pretrained(bert_model)


def preprocess_function(rec):
    """
    Preprocesses a record from the dataset by combining the question title,
    question content, and best answer into a single text string and then
    tokenizing the text string using the tokenizer.

    Arguments:
        rec - record from the dataset
    Returns:
        dictionary containing the tokenized text string
    """
    rec["text"] = rec["question_title"] + " " + \
        rec["question_content"] + " " + rec["best_answer"]
    return tokenizer(rec["text"], truncation=True)


tokenized_yahoo_aswers_topics = yahoo_answers_topics.map(
    preprocess_function).rename_column("topic", "labels")


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

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

id2label = {i: label for i, label in enumerate(YAHOO_CLASSES)}

label2id = {label: i for i, label in enumerate(YAHOO_CLASSES)}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    bert_model, num_labels=10, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="hf_supervised_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_yahoo_aswers_topics["train"].shard(
        index=0, num_shards=2),
    eval_dataset=tokenized_yahoo_aswers_topics["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("hf_supervised_model")

# Test the trained model
text = "What are the elements in water?  Water contains hydrogen and oxygen."

classifier = pipeline("text-classification", model="./hf_supervised_model")
print(classifier(text))
# Display should look like:
# [{'label': 'science or mathematics', 'score': 0.8060687780380249}]
# with score varying depending on the training run
