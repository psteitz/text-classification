from ast import Dict, List
from datasets import load_from_disk
import pandas as pd


TOPIC_LABELS = ["society or culture",
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

def confusion(data_dir: str) -> list:
    """ 
    Load database from disk and iterate to create augmented confusion matrix.

    Returns augmented confusion matrix - a 2d array of lists of record ids from the dataset.
    Rows are predicted labels, columns are correct labels, values are lists of ids in the dataset
    having the corresponding predicted and correct labels. So this is a standard confusion matrix
    whose entries have been exploded into full lists of examples from the data (repreesented by their ids)

    Note: setup assumes that the labels are integers in the range 0 to n-1, where n is the number of labels.
    """
    confusion = []
    ds = load_from_disk(data_dir)
    print("Read ", len(ds), " records from " + data_dir)
    print("First record: " , ds[0])

    # Get list of topics that occur in topic column
    topics = ds.unique("topic")
    topics.sort()
    print("Topics: ", topics)

    # initialize the augmented confusion matrix to be a square matrix of empty lists
    for i in range(len(topics)):
        confusion.append([])
        for j in range(len(topics)):
            confusion[i].append([])
    
    # Iterate the dataset to fill the augmented confusion matrix
    n = len(ds)
    bad_predictions = 0
    for i in range(n):
        prediction = ds[i]["prediction"]
        correct = ds[i]["topic"]
        if prediction == -1:
            bad_predictions += 1
        else:
            confusion[prediction][correct].append(ds[i]["id"])
    return confusion

def metrics(confusion: list) -> Dict:
    """ 
    Calculate metrics from confusion matrix.

    Returns a dictionary of metrics.

    Keys are:
        num_topics - number of topics
        precision - list of precision values in topic order
        recall - list of recall values in topic order
        f1 - list of f1 values in topic order
        macro_precision - average precision
        macro_recall - average recall
        macro_f1 - average f1
        accuracy - overall prediction accuracy
    
    """
    metrics = {}
    num_topics = len(confusion)
    metrics["num_topics"] = num_topics
    metrics["precision"] = []
    metrics["recall"] = []
    metrics["f1"] = []
    metrics["macro_precision"] = 0
    metrics["macro_recall"] = 0
    metrics["macro_f1"] = 0
    correct = 0
    total = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for i in range(num_topics):
        true_positives = len(confusion[i][i])
        correct += true_positives
        false_positives = 0
        false_negatives = 0
        for j in range(num_topics):
            total += len(confusion[i][j])
            if i != j:
                false_positives += len(confusion[i][j])  
                false_negatives += len(confusion[j][i])
        precision = true_positives / (true_positives + false_positives)
        precision_sum += precision
        recall = true_positives / (true_positives + false_negatives)
        recall_sum += recall
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f1_sum += f1
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
    metrics["accuracy"] = correct / total
    metrics["macro_precision"] = precision_sum / metrics["num_topics"]
    metrics["macro_recall"] = recall_sum / metrics["num_topics"]
    metrics["macro_f1"] = f1_sum / metrics["num_topics"]
    return metrics

def show_metrics(metrics: Dict):
    """ 
    Display metrics from confusion matrix.
    """
    print("Number of topics: ", metrics["num_topics"])
    # Loop over topics, displaying topic labels and metrics
    for i in range(metrics["num_topics"]):
        print(i, TOPIC_LABELS[i])
        print("\tPrecision: ", metrics["precision"][i])
        print("\tRecall: ", metrics["recall"][i])
        print("\tF1: ", metrics["f1"][i])
    # Display overall metrics
    print("\nMacro precision: ", metrics["macro_precision"])
    print("Macro recall: ", metrics["macro_recall"])
    print("Macro F1: ", metrics["macro_f1"])
    print("Accuracy: ", metrics["accuracy"])
    
# Demo 
# Load dataset including predictions and topics.  Create augmented confusion matrix from the dataset.
confusion = confusion("../llm/yahoo_answers_topics_augmented")
# Put into a dataframe, replacing lists with counts.  This makes a standard confusion matrix.
frame = pd.DataFrame(confusion)
for i in range(len(frame)):
    frame[i] = frame[i].map(lambda x: len(x))
print(frame)
# Calculate and display metrics from the augmented confusion matrix
print(show_metrics(metrics(confusion)))