#!/usr/bin/env python

"""
train.py: Trains huggingface base transormer model to classify texts in
hf yahoo_answers_topics dataset.
"""
import evaluate
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer, AutoTokenizer, pipeline
from datasets import load_dataset
from train_utils import compute_metrics, preprocess_function

# Huggingace model name for distilbert base uncased
# bert_model = "distilbert-base-uncased"
BERT_MODEL = "bert-base-uncased"

# load HuggingFace dataset
YAHOO_ANSERS_TOPICS = load_dataset(
    "yahoo_answers_topics")

# use AutoTokenizer for BERT_MODEL
TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL)
# truncate at 512 tokens
# leave padding at default
print("Using tokenizer: ", TOKENIZER)

# fine-tuned output model directory
# This directory is created if it doesn't exist and overwritten if it does
OUTPUT_MODEL_DIR = "base_model_from_seg_1"
print("Writing to output model directory: ", OUTPUT_MODEL_DIR)


def preprocess(rec):
    """
    Preprocess a record rec from the dataset.

    Combine the question title, question content, and best answer into a new combined "text" field.

    Add the "text" field to rec.

    Return tokenizer(rec["text"]).

    Arguments:
        rec - record from the dataset (in/out - modified by this function)
        tokenizer - tokenizer to use to tokenize the "text" field

    Returns:
        dictionary containing the tokenized text string
    """
    rec["text"] = rec["question_title"] + " " + \
        rec["question_content"] + " " + rec["best_answer"]
    return TOKENIZER(rec["text"], truncation=True)


TRAIN_DATASET = YAHOO_ANSERS_TOPICS["train"].shard(index=1, num_shards=10).map(
    preprocess).rename_column("topic", "labels")
EVAL_DATASET = YAHOO_ANSERS_TOPICS["test"].map(
    preprocess).rename_column("topic", "labels")


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

ID2LABEL = {i: label for i, label in enumerate(YAHOO_CLASSES)}

LABEL2ID = {label: i for i, label in enumerate(YAHOO_CLASSES)}

MODEL = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL, num_labels=10, id2label=ID2LABEL, label2id=LABEL2ID
)

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
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

TRAINER = Trainer(
    model=MODEL,
    args=TRAINING_ARGS,
    train_dataset=TRAIN_DATASET,
    eval_dataset=EVAL_DATASET,
    tokenizer=TOKENIZER,
    data_collator=DATA_COLLATOR,
    compute_metrics=compute_metrics,
)

TRAINER.train()
TRAINER.save_model(OUTPUT_MODEL_DIR)

"""
Example use of the trained model.
"""
text = "What are the elements in water?  Water contains hydrogen and oxygen."
classifier = pipeline("text-classification", model="./hf_supervised_model")
print(classifier(text))
# Display should look like:
# [{'label': 'science or mathematics', 'score': 0.8060687780380249}]
# with score varying depending on the training run
