from datetime import datetime
import random
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import pipeline, AutoModelForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset

AUGMENTED_DATASET_DIR = "./hf_yahoo_data_augmented"

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

yahoo_answers_topics_augmented = yahoo_anwwers_topics.map(fill_text).remove_columns(["question_title", "question_content", "best_answer"])

def classify_pipe_directly(split: str, num_records : int = -1):
    """
    Pipes the designated split of huggingface yahoo_answers_topics dataset directly
    through the supervised text classification model.

    1. Load the huggingface yahoo_answers_topics dataset.
    2. Augment it with a new column named "text", combining the three text columns in the source dataset into one
    3. Pipe the augmented dataset through the supervised model.
    4. Add model prediction output columns to the dataset.
    5. Save the augmented dataset to AUGMENTED_DATASET_DIR.

    Optional num_records argument limits the number of records read from the dataset. If this argument is present
    and positive, a random sample of num_records records is drawn from the dataset.

    Arguments:
        split - the split of the dataset to load
        num_records - number of records to pipe to the model.  If negative or missing, use all records.
    """
    print("\nclassify_pipe_directly start time: " + datetime.now().strftime("%H:%M:%S"))
    print("\nRunning the dataset directly through the pipline")

    # Load the model
    classifier = pipeline("text-classification", model="./hf_supervised_model", truncation=True, max_length=512, device=0)
    print("Loading and augmenting dataset...")
    ds = yahoo_answers_topics_augmented[split]
    if num_records < 0:
        dataset = ds
    else:
        indices = random.sample(range(0, len(ds)), num_records)
        dataset = ds.select(indices);
    print("Read ", len(dataset), " records from huggingface yahoo_answers_topics " + split + " split.")
    print("First record: ", dataset[0])
    print("Running through pipline...")

    # Create new columns for model output
    labels_column = []
    scores_column = []

    # Run the model on the dataset and fill new columns from model output
    ct = 0
    for out in classifier(KeyDataset(dataset, "sequence")):
        scores_column.append(out["score"])
        labels_column.append(out["label"])
        if ct % 1000 == 0:
            print(str(ct) + "  " + datetime.now().strftime("%H:%M:%S"))
        ct += 1

    # Add columns to dataset
    print ("Adding columns to dataset...")
    print()
    dataset = dataset.add_column("label", labels_column)
    dataset = dataset.add_column("score", scores_column)

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

def score():
    """ 
    Iterate the augmented dataset to compute the loss. 
    Display loss as a proportion.
    """
    ds = load_from_disk(AUGMENTED_DATASET_DIR)
    print("Read ", len(ds), " records from " + AUGMENTED_DATASET_DIR)
    print("First record: " , ds[0])
    
    # Iterate the dataset to compute loss
    loss = 0
    n = len(ds)
    for i in range(n):
        # correct is the value of "topic" in the input dataset
        correct = ds[i]["topic"]
        # predicted is from the model
        predicted = yahoo_index_from_text(ds[i]["label"])
        if not correct == predicted:
            loss += 1
    print ("\nLoss: (number incorrect / number of records)", loss / n)

# Demo
classify_pipe_directly("test")
score()

