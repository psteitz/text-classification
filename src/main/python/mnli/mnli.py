from ast import List
from datetime import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset

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
# Alternative hypothesis templates
DEFAULT_HYPOTHESIS_TYPE = 2
hypothesis_type = DEFAULT_HYPOTHESIS_TYPE
hypothesis_templates = ["This example is {}.",
                        "This text is about {}.",
                        "This question and answer are about {}.",
                        "This question and answer should be classified as {}.",
                        "The right category for this question and answer is '{}'.",
                        "The category of this question and answer is {}.",
                        "This question and answer relates to {}.",
                        "This question question and answer is {}.",
                        "This question is {}.",
                        "This question relates to {}.",
                        "This question is about {}.",
                        "This question should be classified as {}.",
                        "The rigt category for this question is {}.",
                        "The category of this question is {}.",
                        "This text comes under the heading {}."
                        ]

# Facebook BART model fine-tuned for NLI tasks
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', truncation=True)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(device)

def huggingface_zero_shot_classify(sequence, labels : List(str)) -> str:
    """
        Classify the sequence according to the given labels using HuggingFace zero-shot classifier.

        Arguments:
        sequence - text to classify
        hypothesis - classification labels

        Return:
        best classification label

    """
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device=0)
    # 
    # classifer(sequence, labels) returns a dict that looks like:
    # {'sequence': sequence, 
    #  'labels': labels re-ordered so highest probability is first,
    #   'scores': parallel array of probabilities for the re-ordered labels (so highest is first)
    # }
    return classifier(sequence, labels)["labels"][0]   
 

def raw_nli(premise, hypothesis) -> List(float):
    """
        Compute raw NLI entailment logits for the given premise and hypothesis.

        Arguments:
        premise - the premise text
        hypothesis - the hypothesis tex

        Return:
        a list of three logits [contradiction, neutral, entailmenen]

        What is being evaluated is whether or not the premise entails
        (implies) the hypothesis.
    """
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                        truncation=True)
    return nli_model(x.to(device))[0].data.tolist()[0]


# Read in huggingface yahoo_answers_topics dataset and return a list of dicts
def read_huggingface_dataset(split : str) -> List(dict):
    """
    Load the designated split of the huggingface yahoo_answers_topics dataset and return a list of dicts
    with these keys:
        classification  int label index into yahoo_classes
        text  text sequence to be classified
    """
    training_dataset = load_dataset('yahoo_answers_topics', split=split)
    output = []
    for i in range(len(training_dataset)):
        hf_record = training_dataset[i]
        """
        HF yahoo_answers_topics database fields:
           topic (int)	
           question_title (string)	
           question_content (string)
           best_answer (string)
        """
        text = hf_record["question_title"] + " " + hf_record["question_content"] + " " + hf_record["best_answer"]
        output.append({"classification": training_dataset[i]["topic"], "text": text})
    return output

def fill_text(rec):
    """
    Fill in the "text" field of the given record with the concatenation of the three text fields in the record.
    """
    rec["text"] = rec["question_title"] + " " + rec["question_content"] + " " + rec["best_answer"]
    return rec

def read_huggingface_dataset_concat(split : str) -> Dataset:
    """
    Load the designated split of the huggingface yahoo_answers_topics dataset and augment it with a new column named "text",
    combining the three text columns in the source dataset into one.
    """
    raw_dataset = load_dataset('yahoo_answers_topics')
    augmented_dataset = raw_dataset[split]
    new_column = ["null"] * len(augmented_dataset)
    augmented_dataset = augmented_dataset.add_column("text", new_column)
    return augmented_dataset.map(fill_text)

def simple_zero_shot_hf():
    print("HuggingFace zero-shot classifier: ")
    sequence = "What elements make up water? Oygen and hydrogen."
    print("HuggingFace classifier: ", huggingface_zero_shot_classify(sequence, yahoo_classes))

def simple_raw_nli():
    print("Raw NLI logits for each class:")
    sequence = "What elements make up water? Oygen and hydrogen."
    for label in yahoo_classes:
        hypothesis = f'This example is {label}.'
        print(f'Premise: {sequence}')
        print(f'Hypothesis: {hypothesis}')
        print(f'Raw NLI logits: {raw_nli(sequence, hypothesis)}')
        print(f'HuggingFace zero-shot classification: {huggingface_zero_shot_classify(sequence, yahoo_classes)}')
        print()
    print("Full raw output from nli model:")
    hypothesis = 'This example is politics or government.'
    print(f'Premise: {sequence}')
    x = tokenizer.encode(sequence, hypothesis, return_tensors='pt',
                            truncation=True)
    print(nli_model(x.to(device)))

def classify_iterate_hf_dataset():
    print("Read huggingface dataset into list of dicts and start manually iterating and classfiying...")
    records = read_huggingface_dataset("test");
    print("Read ", len(records), " records from huggingface yahoo_answers_topics train split");
    print("First record: ", records[0])
    for i in range(10):
        print(records[i]["classification"], " ", yahoo_classes[records[i]["classification"]])
        print(records[i]["text"])
        print(huggingface_zero_shot_classify(records[i]["text"], yahoo_classes))

def classify_pipe_directly():
    print("Run the dataset directly through the pipline")
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli", device=0)
    print("Loading and augmenting dataset...")
    dataset = read_huggingface_dataset_concat("test")
    print("Read ", len(dataset), " records from huggingface yahoo_answers_topics test split.")
    print("First record: ", dataset[0])
    print("Running through pipline...")

    # Create new columns for model output
    sequence_track_column = [''] * len(dataset)
    labels_column = [['']*len(yahoo_classes)] * len(dataset)
    scores_column = [[0]*len(yahoo_classes)] * len(dataset)

    # Run the model on the dataset and fill new columns from model output
    ct = 0
    for out in classifier(KeyDataset(dataset, "text"), yahoo_classes):
        # print("out = " + out)
        sequence_track_column[ct] = out["sequence"]
        scores_column[ct] = out["scores"]
        labels_column[ct] = out["labels"]
        if ct % 1000 == 0:
            print(str(ct) + "  " + datetime.now().strftime("%H:%M:%S"))
            """
            print("model output")
            print(out)
            print("dataset record")
            print(dataset[ct])
            print()
            """
        ct += 1

    # Add columns to dataset
    dataset = dataset.add_column("labels", labels_column)
    dataset = dataset.add_column("scores", scores_column)
    dataset = dataset.add_column("sequence_track", sequence_track_column)

    # Display the first 5 records of the dataset
    print(dataset[:5])


classify_pipe_directly()
