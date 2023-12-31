from ast import Dict, List
import random
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


def confusion_matrix(dataset_path: str) -> list:
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
        dataset_path - directory containing the dataset
    Returns:
        confusion - augmented confusion matrix as a list of lists of lists of record ids
    """
    confusion = []
    dataset = load_from_disk(dataset_path)
    print("Read ", len(dataset), " records from " + dataset_path)
    print("First record: ", dataset[0])

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
    results = {}
    num_topics = len(confusion)
    results["num_topics"] = num_topics
    results["precision"] = []
    results["recall"] = []
    results["f1"] = []
    results["macro_precision"] = 0
    results["macro_recall"] = 0
    results["macro_f1"] = 0
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
        f1 = 2 * true_positives / \
            (2 * true_positives + false_positives + false_negatives)
        f1_sum += f1
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
    results["accuracy"] = correct / total
    results["macro_precision"] = precision_sum / results["num_topics"]
    results["macro_recall"] = recall_sum / results["num_topics"]
    results["macro_f1"] = f1_sum / results["num_topics"]
    return results


def show_metrics(results: Dict):
    """ 
    Display metrics from confusion matrix.
    """
    print("Number of topics: ", results["num_topics"])
    # Loop over topics, displaying topic labels and metrics
    for i in range(results["num_topics"]):
        print(i, TOPIC_LABELS[i])
        print("\tPrecision: ", results["precision"][i])
        print("\tRecall: ", results["recall"][i])
        print("\tF1: ", results["f1"][i])
    # Display overall metrics
    print("\nMacro precision: ", results["macro_precision"])
    print("Macro recall: ", results["macro_recall"])
    print("Macro F1: ", results["macro_f1"])
    print("Accuracy: ", results["accuracy"])


def examples(dataset: Dataset, predicted: int, correct: int, num_examples: int, confusion: list) -> List:
    """ 
    Return a list of ids of records in the dataset that have the given predicted and correct labels.
    Note: the dataset must have id, prediction, and topic columns defined. 

    Arguments:
        dataset - dataset from which to select examples
        predicted - predicted label
        correct - correct label
        num_examples - maximum number of examples to return (may be less if there are fewer than num_examples examples)
    """
    results = []
    for id in confusion[predicted][correct]:
        results.append(dataset[id])
        if len(results) >= num_examples:
            break
    return results


def mistakes(confusion: List, num_mistakes: int = -1) -> List:
    """ 
    Return list of mistake triples: (predicted, correct, frequency) in descending order of frequency.
    The frequency is the number of times the mistake was made in the dataset
    (the value of <predicted, correct> entry in the confusion matrix).

    Arguments:
        confusion - augmented confusion matrix
        num_mistakes - maximum number of entries to return 
    Returns:
        list of (predicted, correct, frequency) tuples in descending order of frequency 
    """
    results = []
    num_added = 0
    # Iterate over the confusion matrix, adding each off-diagonal entry to the return list
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            if i != j:
                results.append((i, j, len(confusion[i][j])))
                num_added += 1
    # Sort the return list by frequency
    results.sort(key=lambda x: x[2], reverse=True)
    if num_mistakes == -1:
        return results
    return results[:num_mistakes]


def show_mistakes(dataset: Dataset, confusion: List, num_mistakes: int = -1, num_examples: int = 3):
    """ 
    Display the most frequent mistakes in the dataset, along with examples of each mistake.

    Arguments:
        dataset - dataset to search for examples
        confusion - augmented confusion matrix
        num_mistakes - maximum number of mistake entries to display (default is all)
        num_examples - maximum number of examples to display for each mistake (default is 3)
    """
    top_mistakes = mistakes(confusion, num_mistakes)
    for mistake in top_mistakes:
        predicted, correct, frequency = mistake
        print("Predicted:", TOPIC_LABELS[predicted],
              "Correct", TOPIC_LABELS[correct], "count", frequency)
        example_ids = confusion[predicted][correct]
        if len(example_ids) > num_examples:
            # take a random sample of num_examples examples from example_ids
            example_ids = random.sample(example_ids, num_examples)
            num_examples = len(example_ids)

        for i in range(num_examples):
            # print("\t", example_ids[i])
            # find the example in the dataset and print the value of its sequence column
            for j, rec in enumerate(dataset):
                if rec['id'] == example_ids[i]:
                    break
            print(dataset[j]['text'], '\n')

# Demo
# Load dataset including predictions and topics.  Create augmented confusion matrix from the dataset.


# confusion = confusion("../supervised/data/hf_yahoo_data_augmented")
# confusion = confusion("../mnli/hf_yahoo_data_augmented")
base_data_dir = "../supervised/data/hf_yahoo_data_augmented_base"
base_confusion_matrix = confusion_matrix(base_data_dir)
# Put into a dataframe, replacing lists with counts.  This makes a standard confusion matrix.
base_frame = pd.DataFrame(base_confusion_matrix)
for i in range(len(base_frame)):
    base_frame[i] = base_frame[i].map(lambda x: len(x))
print("Confusion matrix: rows are predicted labels, columns are correct labels")
print(base_frame)
# Calculate and display metrics from the augmented confusion matrix
print("Base model metrics:")
show_metrics(metrics(base_confusion_matrix))
# show_mistakes(load_from_disk(data_dir), confusion_matrix, 10)

student_data_dir = "../supervised/data/hf_yahoo_data_augmented_student"
student_confusion_matrix = confusion_matrix(student_data_dir)
# Put into a dataframe, replacing lists with counts.  This makes a standard confusion matrix.
student_frame = pd.DataFrame(student_confusion_matrix)
for i in range(len(student_frame)):
    student_frame[i] = student_frame[i].map(lambda x: len(x))
print("Confusion matrix: rows are predicted labels, columns are correct labels")
print(student_frame)
# Calculate and display metrics from the augmented confusion matrix
print("Student model metrics:")
show_metrics(metrics(student_confusion_matrix))
# show_mistakes(load_from_disk(data_dir), confusion_matrix, 10)
