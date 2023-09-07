from datetime import datetime
import random
from pathlib import Path

from torch import tensor
import torch
from datasets import load_dataset, load_from_disk
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset

AUGMENTED_DATASET_DIR = "./hf_yahoo_data_augmented"
FULLY_AUGMENTED_DATASET_DIR = "./hf_yahoo_data_fully_augmented"

# yahoo questions categories
yahoo_classes = [
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

yahoo_anwwers_topics = load_dataset("yahoo_answers_topics")

def fill_text(rec):
    """
    Fill in the "sequence" field of the given record with the concatenation of the three text fields in the record.
    """
    rec["sequence"] = rec["question_title"] + " " + rec["question_content"] + " " + rec["best_answer"]
    return rec

yahoo_answers_topics_augmented = yahoo_anwwers_topics.map(fill_text)\
    .remove_columns(["question_title", "question_content", "best_answer"])

def classify(split: str, num_records : int = -1):
    """
    Pipes the designated split of huggingface yahoo_answers_topics dataset through the supervised text classification model.
    Augments the dataset with predictions and saves the augmented dataset to disk.

    1. Load the huggingface yahoo_answers_topics dataset.
    2. Augment it with a new column named "text", combining the three text columns in the source dataset into one
    3. Pipe the augmented dataset through the supervised model.
    4. Add model prediction output columns to the dataset.
    5. Save the augmented dataset to AUGMENTED_DATASET_DIR.

    Optional num_records argument limits the number of records read from the dataset. If this argument is present
    and positive, a random sample of num_records records is drawn from the dataset.

    Arguments:
        split - the split of the dataset to load - "train", "validation", or "test"
        num_records - number of records to pipe to the model.  If negative or missing, use all records.
    """
    print("\nclassify_pipe_directly start time: " + datetime.now().strftime("%H:%M:%S"))
    print("\nRunning the dataset directly through the pipline")

    # Load the model
    classifier = pipeline("text-classification", model="./hf_supervised_model", truncation=True, max_length=512, device=0)
    print("Loading and augmenting dataset...")
    dataset = yahoo_answers_topics_augmented[split]
    if num_records < 0:
        dataset = dataset
    else:
        indices = random.sample(range(0, len(dataset)), num_records)
        dataset = dataset.select(indices);
    print("Read ", len(dataset), " records from huggingface yahoo_answers_topics " + split + " split.")
    print("First record: ", dataset[0])
    print("Running through pipline...")

    # Create new columns for model output
    labels_column = []
    scores_column = []
    predictions_column = []

    # Run the model on the dataset and fill new columns from model output
    # Model output is a dictionary with keys "score" and "label" where "label"
    # is the predicted class label and "score" is the softmax score for the prediction.
    ct = 0
    for out in classifier(KeyDataset(dataset, "sequence")):
        scores_column.append(out["score"])
        labels_column.append(out["label"])
        predictions_column.append(yahoo_index_from_text(out["label"]))
        if ct % 1000 == 0:
            print(str(ct) + "  " + datetime.now().strftime("%H:%M:%S"))
        ct += 1

    # Add columns to dataset
    print ("Adding columns to dataset...")
    print() 

    dataset = dataset.add_column("label", labels_column)\
        .add_column("score", scores_column).\
            add_column("prediction", predictions_column)

    # Display the first record  of the dataset
    print(dataset[0])

    print("Saving augmented dataset to disk...")
    # Save the augmented dataset to disk
    Path(AUGMENTED_DATASET_DIR).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(AUGMENTED_DATASET_DIR)
    print("\nclassify_pipe_directly end time: " + datetime.now().strftime("%H:%M:%S"))

def yahoo_index_from_text(class_label: str) -> int:
    """
    Return the index of class_label in yahoo_classes.
    """
    for i in range(0, len(yahoo_classes)):
        if class_label == yahoo_classes[i]:
            return i 
    print("Error: yahoo class not found: " + class_label)
    return -1



# Load the model and tokenizer
MODEL = AutoModelForSequenceClassification.from_pretrained("./hf_supervised_model")
TOKENIZER = AutoTokenizer.from_pretrained("./hf_supervised_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)

def fill_output(rec):
    """
    Fill in the "text" field of the given record with the concatenation of the three text fields in the record.
    """

    rec["text"] = rec["question_title"] + " " + rec["question_content"] + " " + rec["best_answer"]
    return rec

def fill_model_output(rec):
    inputs = TOKENIZER(rec["sequence"], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    model_output = MODEL(**inputs, output_hidden_states=True)
    final_encoding = model_output.hidden_states[0][:, 0, :] 
    logits = model_output.logits
    rec["embeddings"] = final_encoding
    rec["logits"] = logits
    return rec

def add_model_output():
    """
    Add final embeddings and output logits to the augmented dataset.
    Embeddings are the final hidden state of the model. 
    Output columns are set by mapping fill_model_output over the augmented dataset.

    Note: AUGMENTED_DATASET_DIR must already exist.  
    This function does not create it and it does not add predictions.
    Run classify("test") to create the augmented dataset using the test split
    """
    print("\add_model_output start time: " + datetime.now().strftime("%H:%M:%S"))
    print("\nLoading augmented dataset from disk...")
    dataset = load_from_disk(AUGMENTED_DATASET_DIR)
    print("Read ", len(dataset), " records from augmented dataset.")
    print("First record: ", dataset[0])
    # Create new columns for embeddings and logits from the model
    # Embeddings are the final hidden state of the model.
    embeddings_column = [None] * len(dataset)
    # Create a new column for model output logits - the raw output of the model before softmax
    logits_column = [None] * len(dataset)
    # Add the new columns to the dataset
    dataset = dataset.add_column("embeddings", embeddings_column)
    dataset = dataset.add_column("logits", logits_column)
    print("Running the model over the dataset...")
    # Map fill_model_output over the dataset
    dataset = dataset.map(fill_model_output, writer_batch_size=10)
    # Save the augmented dataset to disk
    print("Saving fully augmented dataset to disk...")
    dataset.save_to_disk(FULLY_AUGMENTED_DATASET_DIR)
    print("\add_model_output end time: " + datetime.now().strftime("%H:%M:%S"))
# Demo
# classify("test")
add_model_output()

