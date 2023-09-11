from ast import Dict, List
from datasets import load_from_disk, Dataset
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
    Load database from disk and create augmented confusion matrix.

    The augmented confusion matrix is a 2d array whose elements are lists of record ids from the dataset.
    Rows are predicted labels, columns are correct labels, values are lists of ids in the dataset
    having the corresponding predicted and correct labels. So this is a standard confusion matrix
    whose entries have been exploded into full lists of examples from the data (repreesented by their ids).
    For example, if the confusion matrix has [12, 93, 2005] in row 3, column 5, then the records with ids
    12, 93, and 2005 in the dataset have predicted label 3 and correct label 5.

    Note: setup assumes that the labels are integers in the range 0 to n-1, where n is the number of labels.

    Arguments:
        data_dir - directory containing the dataset
    Returns:
        confusion - augmented confusion matrix as a list of lists of lists of record ids
    """
    confusion = []
    dataset = load_from_disk(data_dir)
    print("Read ", len(dataset), " records from " + data_dir)
    print("First record: " , dataset[0])

    # Get list of topics that occur in topic column
    # This should be 0 to n-1, where n is the number of topics
    topics = dataset.unique("topic")
    topics.sort()

    # initialize the augmented confusion matrix to be a square matrix of empty lists
    for row in range(len(topics)):
        confusion.append([])
        for j in range(len(topics)):
            confusion[row].append([])
    
    # Iterate the dataset to fill the augmented confusion matrix
    num_rows = len(dataset)
    bad_predictions = 0
    for row in range(num_rows):
        prediction = dataset[row]["prediction"]
        correct = dataset[row]["topic"]
        if prediction == -1:
            bad_predictions += 1
        else:
            confusion[prediction][correct].append(dataset[row]["id"])
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
    
    Arguments:
        confusion - augmented confusion matrix
        Augmented means the elements are full lists of record ids instead of counts. 
        Rows are predicted labels, columns are correct labels, values are lists of ids in the dataset that have
        the corresponding predicted and correct labels. 
    
    Returns:
        metrics - metrics dictionary 
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

def examples(dataset: Dataset, predicted : int, correct : int, num_examples : int, confusion : list) -> List:
    """ 
    Return a list of ids of records in the dataset that have the given predicted and correct labels.
    Note: the dataset must have id, prediction, and topic columns defined. 

    Arguments:
        dataset - dataset from which to select examples
        predicted - predicted label
        correct - correct label
        num_examples - maximum number of examples to return (may be less if there are fewer than num_examples examples)
    """
    examples = []
    for id in confusion[predicted][correct]:
        examples.append(dataset[id])
        if len(examples) >= num_examples:
            break
    return examples
    
def mistakes(num_mistakes : int = -1) -> List:
    """ 
    Return list of mistake triples: (predicted, correct, frequency) in descending order of frequency.
    The frequency is the number of times the mistake was made in the dataset
    (the value of <predicted, correct> entry in the confusion matrix).
    
    Arguments:
        num_mistakes - maximum number of entries to return 
    Returns:
        list of (predicted, correct, frequency) tuples in descending order of frequency 
    """
    # Load the confusion matrix and create the retun list   
    augmented_confusion = confusion("../llm/data/yahoo_answers_topics_augmented_zero_shot")
    mistakes = []
    num_added = 0
    # Iterate over the confusion matrix, adding each off-diagonal entry to the return list
    for i in range(len(augmented_confusion)):
        for j in range(len(augmented_confusion)):
            if i != j:
                mistakes.append((i, j, len(augmented_confusion[i][j])))
                num_added += 1
    # Sort the return list by frequency
    mistakes.sort(key=lambda x: x[2], reverse=True)
    if num_mistakes == -1:
        return mistakes
    return mistakes[:num_mistakes]

def show_mistakes(num_mistakes : int = -1):
    """ 
    Display the most frequent mistakes in the dataset.
    
    Arguments:
        num_mistakes - maximum number of entries to display 
    """
    top_mistakes = mistakes(num_mistakes)
    for mistake in top_mistakes:
        predicted, correct, frequency = mistake
        print("Predicted:",TOPIC_LABELS[predicted], "Correct", TOPIC_LABELS[correct], "count", frequency)

# Demo 
# Load dataset including predictions and topics.  Create augmented confusion matrix from the dataset.
#confusion = confusion("../supervised/data/hf_yahoo_data_augmented")
#confusion = confusion("../mnli/hf_yahoo_data_augmented")
confusion_matrix = confusion("../llm/data/yahoo_answers_topics_augmented_zero_shot")
# Put into a dataframe, replacing lists with counts.  This makes a standard confusion matrix.
frame = pd.DataFrame(confusion_matrix)
for i in range(len(frame)):
    frame[i] = frame[i].map(lambda x: len(x))
print("Confusion matrix: rows are predicted labels, columns are correct labels")
print(frame)
# Calculate and display metrics from the augmented confusion matrix
show_metrics(metrics(confusion_matrix))
show_mistakes(10)