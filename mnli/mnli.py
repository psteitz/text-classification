
from ast import List
from datetime import datetime
from pathlib import Path
import sys
import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset, load_from_disk

# Allow peer module imports
sys.path.append("..")
from constants import defaults  # noqa: E402

# Facebook BART model fine-tuned for NLI tasks
nli_model = AutoModelForSequenceClassification.from_pretrained(
    defaults.HF_MNLI, truncation=True)
tokenizer = AutoTokenizer.from_pretrained(defaults.HF_MNLI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(device)


def fix_seed(seed: int = defaults.DEFAULT_SEED, with_cuda: bool = False):
    """
    Seed for the random number generator.

    Setting a fixed seed will make sampling-based results repeatable.
    By default, only the python random number generator is seeded.
    Set with_cuda=True to also seed the CUDA and torc generator.

    Arguments:
        seed - the seed value to use
        with_cuda - if True, also seed the CUDA and torch generators
    """
    random.seed(seed)
    if with_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def hf_zero_shot_classify(sequence, labels: List(str)) -> str:
    """
        Classify the sequence according to the given labels using HuggingFace zero-shot classifier.

        Arguments:
          sequence  text to classify
          labels    array of classification labels

        Return:
        best classification label

    """
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli", device=0)
    #
    # classifer(sequence, labels) returns a dict that looks like:
    #
    # {'sequence': sequence,
    #  'labels': labels re-ordered so highest probability is first,
    #  'scores': probabilities for the re-ordered labels (so highest is first)
    # }
    return classifier(sequence, labels)["labels"][0]


def raw_nli(premise, hypothesis) -> List(float):
    """
        Compute raw NLI entailment logits for the given premise and hypothesis.

        Arguments:
        premise - the premise text
        hypothesis - the hypothesis text

        Return:
        a list of three logits [contradiction, neutral, entailment]

        What is being evaluated is whether or not the premise entails
        (implies) the hypothesis.
    """
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                         truncation=True)
    return nli_model(x.to(device))[0].data.tolist()[0]


def fill_text(rec):
    """
    Fill in the "text" field of the given record with the concatenation of the three text fields in the record.
    """
    rec["text"] = rec["question_title"] + " " + \
        rec["question_content"] + " " + rec["best_answer"]
    return rec


def augment_huggingface_dataset(split: str) -> Dataset:
    """
    Load the designated split of the huggingface yahoo_answers_topics dataset and augment it with a new column named "text",
    combining the three text columns in the source dataset into one. Then drop the three source text columns. Return the
    augmented dataset.

    Arguments:
        split - the split of the dataset to load

    Return:
        the augmented dataset
    """
    raw_dataset = load_dataset('yahoo_answers_topics')
    augmented_dataset = raw_dataset[split]
    new_column = ["null"] * len(augmented_dataset)
    augmented_dataset = augmented_dataset.add_column("text", new_column)
    augmented_dataset = augmented_dataset.map(fill_text)
    augmented_dataset.remove_columns(
        ["question_title", "question_content", "best_answer"])
    return augmented_dataset


def simple_zero_shot_hf():
    print("HuggingFace zero-shot classifier: ")
    sequence = "What elements make up water? Oygen and hydrogen."
    print("HuggingFace classifier: ", hf_zero_shot_classify(
        sequence, defaults.YAHOO_CLASSES))


def simple_raw_nli():
    print("Raw NLI logits for each class:")
    sequence = "What elements make up water? Oygen and hydrogen."
    for label in defaults.YAHOO_CLASSES:
        hypothesis = f'This example is {label}.'
        print(f'Premise: {sequence}')
        print(f'Hypothesis: {hypothesis}')
        print(f'Raw NLI logits: {raw_nli(sequence, hypothesis)}')
        print(
            f'HuggingFace zero-shot classification: {hf_zero_shot_classify(sequence, defaults.YAHOO_CLASSES)}')
        print()
    print("Full raw output from nli model:")
    hypothesis = 'This example is politics or government.'
    print(f'Premise: {sequence}')
    x = tokenizer.encode(sequence, hypothesis, return_tensors='pt',
                         truncation=True)
    print(nli_model(x.to(device)))


def classify_pipe_directly(split: str, num_records: int = -1):
    """
    Pipes the designated split of huggingface yahoo_answers_topics dataset directly
    through the zero-shot-classification model. Saves augmented dataset inluding predictions to disk.

    1. Load the huggingface yahoo_answers_topics dataset.
    2. Augment it with a new column named "text", combining the three text columns in the source dataset into one
    3. Pipe the augmented dataset through the zero-shot-classification model.
    4. Add model prediction output columns to the dataset.
    5. Save the augmented dataset to AUGMENTED_DATASET_DIR.

    Optional num_records argument limits the number of records read from the dataset. If this argument is present
    and positive, a random sample of num_records records is drawn from the dataset.

    Arguments:
        split - the split of the dataset to load
        num_records - number of records to pipe to the model.  If negative or missing, use all records.
    """
    print("\nclassify_pipe_directly start time: " +
          datetime.now().strftime("%H:%M:%S"))
    print("\nRunning the dataset directly through the pipline")

    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli", device=0)
    print("Loading and augmenting dataset...")
    ds = augment_huggingface_dataset(split)
    if num_records < 0:
        dataset = ds
    else:
        indices = random.sample(range(0, len(ds)), num_records)
        dataset = ds.select(indices)
    print("Read ", len(dataset),
          " records from huggingface yahoo_answers_topics " + split + " split.")
    print("First record: ", dataset[0])
    print("Running through pipline...")

    # Create new columns for model output
    labels_column = []  # list of lists of labels, ordred by probability
    scores_column = []  # list of lists of scores, ordred by probability
    # list of predicted labels (index of the first label in labels_column)
    predictions_column = []

    # Add a column to track hash of text
    sequence_track_column = []

    # Run the model on the dataset and fill new columns from model output
    ct = 0
    for out in classifier(KeyDataset(dataset, "text"), defaults.YAHOO_CLASSES):
        sequence_track_column.append(hash(out["sequence"]))
        scores_column.append(out["scores"])
        labels_column.append(out["labels"])
        predictions_column.append(yahoo_index_from_text(out["labels"][0]))
        if ct % 1000 == 0:
            print(str(ct) + "  " + datetime.now().strftime("%H:%M:%S"))
        ct += 1

    # Add columns to dataset
    print("Adding columns to dataset...")
    print()
    dataset = dataset.add_column("labels", labels_column)\
        .add_column("scores", scores_column)\
        .add_column("sequence_track", sequence_track_column)\
        .add_column("prediction", predictions_column)

    # Display the first record  of the dataset
    print(dataset[0])

    # Save the augmented dataset to disk
    print("Saving augmented dataset to disk...")
    Path(defaults.AUGMENTED_DATASET_DIR).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(defaults.AUGMENTED_DATASET_DIR)
    print("\nclassify_pipe_directly end time: " +
          datetime.now().strftime("%H:%M:%S"))


def yahoo_index_from_text(class_label: str) -> int:
    """
    Return the index of class_label in yahoo_classes.
    """
    for i in range(0, len(defaults.YAHOO_CLASSES)):
        if class_label == defaults.YAHOO_CLASSES[i]:
            return i
    print("Error: yahoo class not found: " + class_label)
    return -1


def check_hashes():
    """
    Load the augmented dataset from disk.
    Iterate over the dataset, verifying that the hash of the text matches the hash of the sequence.
    """
    ds = load_from_disk(defaults.AUGMENTED_DATASET_DIR)
    print("Read ", len(ds), " records from " + defaults.AUGMENTED_DATASET_DIR)
    for i in range(0, len(ds)):
        if hash(ds[i]["text"]) != ds[i]["sequence_track"]:
            print("Error: hash mismatch at index " + str(i))
            print("Text: ", ds[i]["text"])
            print("Sequence: ", ds[i]["sequence"])
            print("Hash of text: ", hash(ds[i]["text"]))
            print("Hash of sequence: ", ds[i]["sequence_track"])
            return
    print("All hashes match!")


# Demo
# fix_seed()
# classify_pipe_directly("test")
# check_hashes()
