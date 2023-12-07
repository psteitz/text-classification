"""
Utilities used by trainers.
"""
import evaluate  # huggingface evaluate library
import numpy as np

ACCURACY = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = ACCURACY.compute(predictions=predictions, references=labels)
    print("compute_metrics: result = ", result)
    return result


def preprocess_function(rec, tokenizer):
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
    return tokenizer(rec["text"], truncation=True)
