"""
Inference class: encapsulates the supervised model and provides methods for inference.
"""
from datetime import datetime
import random
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset

# yahoo questions categories
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


class Inference:

    def __init__(self, augmented_dataset_dir="./data//hf_yahoo_data_augmented",
                 fully_augmented_dataset_dir="./data/hf_yahoo_data_fully_augmented",
                 model_directory="./hf_supervised_model_0",
                 hf_dataset_name="yahoo_answers_topics"):
        """
        Inference: initialize the inference object.

        Arguments:
            augmented_dataset_dir - the directory to store the augmented dataset

            fully_augmented_dataset_dir - the directory to store the fully augmented dataset

            model_directory - the directory containing the model

            hf_dataset_name - the name of the huggingface dataset to load
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_directory)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model.to(self.device)
        self.augmented_dataset_dir = augmented_dataset_dir
        self.fully_augmented_dataset_dir = fully_augmented_dataset_dir
        self.model_directory = model_directory
        self.hf_dataset_name = hf_dataset_name

    def yahoo_index_from_text(self, class_label: str) -> int:
        """
        yahoo_index_from_text: return the index of class_label in yahoo_classes.

        Arguments:
        class_label - the class label to look up
        Returns:
            the index of class_label in yahoo_classes
        """
        for i, label in enumerate(YAHOO_CLASSES):
            if class_label == label:
                return i
        print("Error: yahoo class not found: " + class_label)
        return -1

    def fill_text(self, rec):
        """
        Fill in the "sequence" field of the given record with the concatenation of the three text fields in the record.
        """
        rec["sequence"] = rec["question_title"] + " " + \
            rec["question_content"] + " " + rec["best_answer"]
        return rec

    def classify_tokenized(self, inputs) -> dict:
        """
        Classify the given tokenized text using the model.
        """
        with torch.no_grad():
            output = {}
            logits = self.model(**inputs).logits
            predicted_distribution = torch.nn.Softmax(
                dim=-1)(logits).tolist()[0]
            for i, prob in enumerate(predicted_distribution):
                output[YAHOO_CLASSES[i]] = prob
            return output

    def classify(self, text) -> dict:
        """
        Classify the given text using the model.

        Arguments:
            text - the text to classify
            model - the model to use for classification
        Returns:
            a dictionary with keys "score" and "label" where "label" is the predicted class label and
            "score" is the softmax score for the prediction.
        """
        return self.classify_tokenized(self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device))

    def classify_dataset(self, split: str, num_records: int = -1):
        """
        Pipes the designated split of huggingface yahoo_answers_topics dataset through the supervised text classification model.
        Augments the dataset with predictions and saves the augmented dataset to disk.

        1. Load the huggingface yahoo_answers_topics dataset.
        2. Augment it with a new column named "text", combining the three text columns in the source dataset into one
        3. Pipe the augmented dataset through the supervised model.
        4. Add model prediction output columns to the dataset.
        5. Save the augmented dataset to augmented_dataset_dir.

        Optional num_records argument limits the number of records read from the dataset. If this argument is present
        and positive, a random sample of num_records records is drawn from the dataset.

        Arguments:
            split - the split of the dataset to load - "train", "validation", or "test"
            num_records - number of records to pipe to the model.  If negative or missing, use all records.
        """
        print("\nclassify_pipe_directly start time: " +
              datetime.now().strftime("%H:%M:%S"))
        print("\nRunning the dataset directly through the pipline")

        # Load the model
        classifier = pipeline("text-classification", model=self.model_directory,
                              truncation=True, max_length=512, device=0)
        print("Loading and augmenting dataset...")
        yahoo_anwwers_topics = load_dataset(self.hf_dataset_name)
        yahoo_answers_topics_augmented = yahoo_anwwers_topics.map(self.fill_text)\
            .remove_columns(["question_title", "question_content", "best_answer"])
        dataset = yahoo_answers_topics_augmented[split]
        if num_records > 0:
            indices = random.sample(range(0, len(dataset)), num_records)
            dataset = dataset.select(indices)
        print("Read ", len(dataset),
              " records from huggingface dataset", self.hf_dataset_name, " ", split, " split.")
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
            predictions_column.append(self.yahoo_index_from_text(out["label"]))
            if ct % 1000 == 0:
                print(str(ct) + "  " + datetime.now().strftime("%H:%M:%S"))
            ct += 1

        # Add columns to dataset
        print("Adding columns to dataset...")
        print()

        dataset = dataset.add_column("label", labels_column)\
            .add_column("score", scores_column).\
            add_column("prediction", predictions_column)

        # Display the first record  of the dataset
        print(dataset[0])

        print("Saving augmented dataset to disk...")
        # Save the augmented dataset to disk
        Path(self.augmented_dataset_dir).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(self.augmented_dataset_dir)
        print("\nclassify_pipe_directly end time: " +
              datetime.now().strftime("%H:%M:%S"))

    def predict(self, rec: dict) -> dict:
        """
        Predict the class label for the given record using the model.
        Add model output (raw logits and final hidden states) to the record
        and return the augmented record.

        Arguments:
            rec - the record to classify based on the contents of its "sequence" field
        Returns:
            the record with model output added
        """
        inputs = self.tokenizer(
            rec["sequence"], return_tensors="pt", truncation=True, max_length=512).to(self.device)
        model_output = self.model(**inputs, output_hidden_states=True)
        final_encoding = model_output.hidden_states[0][:, 0, :]
        logits = model_output.logits
        rec["embeddings"] = final_encoding
        rec["logits"] = logits
        return rec

    def add_model_output(self):
        """
        Add final embeddings and raw output logits to the augmented dataset.
        Embeddings are the final hidden states of the model. 
        Output columns are set by mapping predict over the augmented dataset.
        The augmented dataset is saved to self.fully_augmented_dataset_dir.

        Note: self.augmented_dataset_dir must already exist.  
        This function does not create it and it does not add predictions.
        Run classify_dataset("test") first to create the augmented dataset.
        """
        print("\add_model_output start time: " +
              datetime.now().strftime("%H:%M:%S"))
        print("\nLoading augmented dataset from disk...")
        dataset = load_from_disk(self.augmented_dataset_dir)
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
        dataset = dataset.map(self.predict, writer_batch_size=10)
        # Save the augmented dataset to disk
        print("Saving fully augmented dataset to disk...")
        dataset.save_to_disk(self.fully_augmented_dataset_dir)
        print("\add_model_output end time: " +
              datetime.now().strftime("%H:%M:%S"))


# Demo
#
# Use default datasets:
#   augmented_dataset_dir = "./data//hf_yahoo_data_augmented"
#   fully_augmented_dataset_dir = "./data/hf_yahoo_data_fully_augmented"
#   model_directory = "./hf_supervised_model"
#   ^^^ needs to point to a pre-trained supervised model
#

inference_base = Inference("./data//hf_yahoo_data_augmented_base",
                           "./data/hf_yahoo_data_fully_augmented",
                           "./base_model_from_seg_1",
                           "yahoo_answers_topics")
inference_base.classify_dataset("test")

inference_student = Inference("./data//hf_yahoo_data_augmented_student",
                              "./data/hf_yahoo_data_fully_augmented",
                              "./student_model_0",
                              "yahoo_answers_topics")
inference_student.classify_dataset("test")

"""
print("classify method demo:")
inference = Inference()
print(inference.classify("How good are the Jets? I mean the football team. They suck."))

print("predict method demo:")
rec = {"sequence": "What was Frank Sinatra's greatest song? My Way."}
print(inference.predict(rec)['logits'][0].tolist())
"""
