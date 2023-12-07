#!/usr/bin/env python

"""
train_incremental.py: Implements incremental training to train a supervised text classifier.
"""

import evaluate
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments, AutoTokenizer, pipeline
from datasets import load_dataset
from StudentTrainer import StudentTrainer
from train_utils import compute_metrics, preprocess_function

# Directory where the teacher model is saved
TEACHER_MODEL_DIR = "base_model"

# Directory where the student model will be saved
STUDENT_MODEL_DIR = "student_model_0"

# Huggingace Bert model name for distilbert base uncased
BERT_MODEL = "bert-base-uncased"

# load HuggingFace dataset
YAHOO_ANSWERS_TOPICS = load_dataset("yahoo_answers_topics")

# use AutoTokenizer for BERT_MODEL
TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL)

# Amount that teacher model correct answer has to win by in order for student to use it
THRESHOLD = 0.4


TOKENIZED_YAHOO_ANSWERS_TOPICS = YAHOO_ANSWERS_TOPICS.map(
    preprocess_function).rename_column("topic", "labels")

TRAIN_DATASET = TOKENIZED_YAHOO_ANSWERS_TOPICS["train"].shard(
    index=1, num_shards=10)


DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)

ACCURACY = evaluate.load("accuracy")

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

BASE_MODEL = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL, num_labels=10, id2label=id2label, label2id=label2id
)

TRAINING_ARGS = TrainingArguments(
    output_dir=STUDENT_MODEL_DIR,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

TRAINER = StudentTrainer(
    model=BASE_MODEL,
    args=TRAINING_ARGS,
    train_dataset=TRAIN_DATASET,
    eval_dataset=TOKENIZED_YAHOO_ANSWERS_TOPICS["test"],
    tokenizer=TOKENIZER,
    data_collator=DATA_COLLATOR,
    compute_metrics=compute_metrics,
    teacher_model_dir=TEACHER_MODEL_DIR,
    threshhold=THRESHOLD,
)

TRAINER.train()
TRAINER.save_model(STUDENT_MODEL_DIR)

# Test the trained model
TEXT = "What are the elements in water?  Water contains hydrogen and oxygen."

CLASSIFIER = pipeline("text-classification",
                      model=STUDENT_MODEL_DIR)
print(CLASSIFIER(TEXT))
# Display should look like:
# [{'label': 'science or mathematics', 'score': 0.8060687780380249}]
# with score varying depending on the training run
