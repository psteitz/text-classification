"""
Zero-shot classification example using llama-2-70b-chat accessed via the Replicate API

Requires pip install replicate
Export Replicate API key as REPLICATE_API_TOKEN.
To get a key:  https://replicate.com.
Free usage is limited.
"""

from ast import Dict, List
import random
import time
from datasets import load_dataset, Dataset, load_from_disk
import replicate


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

# Directory where augmented datasets are stored
AUGMENTED_DATASET_DIR = "yahoo_answers_topics_augmented"

# Replicate model name
REPLICATE_MODEL_NAME = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"

# Huggingface model name
HF_MODEL_NAME = "microsoft/DialoGPT-medium"

# Number of examples to hold for each prediction error
MAX_ERROR_EXAMPLES = 5

DEFAULT_MODEL_NAME = REPLICATE_MODEL_NAME


categories_string = ""
for item in yahoo_classes:
    categories_string += item + "\n"

system_prompt = "Your job is to find the right category for each question and answer pair provided.\n\
The categories are: \n" + categories_string + "\n Please do not repeat the question in your response or \
add any explanation for your answer. Just respond with the best category name.\nFor example: 'family or relationships'.\n\
Please respond with the category name only.  Do not include any other text in your response and make sure that your response \
matches one of the categories exactly."
                
def classify(question_anawer : str) -> str:
   """
    Classify the given question/answer pair.
   """
   model_response = replicate.run(
        REPLICATE_MODEL_NAME,
        input={"system_prompt": system_prompt, "prompt": question_anawer}
    )
   # Consume the model response stream and build the output string from it.
   output = ""
   for item in model_response:
        output += item
   return output


def yahoo_index_from_text(text: str) -> int:
    """
    Return the index of text in yahoo_classes.

    Returns the index of the first class in yahoo_classes
    that is a substring of text.lower(); -1 if there is no such class.

    Arguments:
        text - input text
    Returns:
        the index of the first class in yahoo_classes that is a substring of text.lower();
       -1 if there is no such class.
    """
    lower = text.lower()

    for i in range(0, len(yahoo_classes)):
        if yahoo_classes[i] in lower:
            return i 
    print("Error: yahoo class not found: " + text)
    return -1


def fill_text(rec):
    """
    Fill in the "text" field of the given record with the concatenation of the three text fields in the record.
    """
    rec["text"] = rec["question_title"] + " " + rec["question_content"] + " " + rec["best_answer"]
    return rec

def prepare_dataset() -> Dataset:
    """
    Preprocceses the yahoo_answers_topics dataset, concatenating the three text columns
    into a single new column called "text".  Removes the three original text columns.
    Also adds a new column called "prediction" to be filled with model responses.
    """
    dataset = load_dataset('yahoo_answers_topics', split='test')
    text_column = ["null"] * len(dataset)
    prediction_column = ["null"] * len(dataset)
    dataset = dataset.add_column("text", text_column)
    dataset = dataset.add_column("prediction", prediction_column)
    return dataset.map(fill_text).remove_columns(["question_title", "question_content", "best_answer"])

def add_prediction(dataset, predictions, responses, i):
    """
    Make prediction based on "text" field and store prediction and raw model response 
    for the given record to the predictions and responses columns, resp.

    Arguments:
        dataset - the dataset 
        predictions - the predictions column
        responses - the responses column
        i - the index of the record to process
    """
    text = dataset[i]["text"]
    model_response = classify(text)
    responses[i] = model_response
    predictions[i] = yahoo_index_from_text(model_response)

    print("Prediction: ", predictions[i])
    print("Text: ", text)

def make_predictions(dataset: Dataset, max_records : int = -1):
    """
    Make predictions for the given dataset and save the dataset,
    augmented with predictions and model response text.

    Arguments:
        dataset - the dataset to augment with predictions based on "text" column
        max_records - if this is positive and smaller than len(dataset), 
                      sample max_records records from dataset
    """
    if max_records > 0 and max_records < len(dataset):
        indices = random.sample(range(0, len(dataset)), max_records)
        dataset = dataset.select(indices);
    predictions = [-1] * len(dataset)
    responses = ["null"] * len(dataset)
    num_predictions = 0
    for i in range(0, len(dataset)):
        add_prediction(dataset, predictions, responses, i)
        num_predictions += 1
        if num_predictions % 5 == 0:
            print("Predicted ", num_predictions, " records.")
            print("Saving augmented dataset to disk.")
            # Drop the previous running columns if they exist
            if "prediction" in dataset.column_names:
                dataset = dataset.remove_columns(["prediction"])
            if "response" in dataset.column_names:
                dataset = dataset.remove_columns(["response"])
            # Add copies of current predictions and responses columns
            dataset = dataset.add_column("prediction", predictions.copy())
            dataset = dataset.add_column("response", responses.copy())
            # Save to disk
            dataset.save_to_disk(AUGMENTED_DATASET_DIR)
            print("Saved augmented dataset to disk.")
            print("Last record saved: ", dataset[i])
        # Wait 500 ms before submitting next prediction request
        time.sleep(0.5)
    print("Final save of augmented dataset to disk.")
    dataset.save_to_disk(AUGMENTED_DATASET_DIR)
    print("Last record: ", dataset[len(dataset) - 1])

def prediction_error_exists(prediction, label, prediction_errors) -> bool:
    """
    Return True if prediction_errors includes a record with the given prediction and label.
    """
    for prediction_error in prediction_errors:
        if prediction_error["predicted_label"] == prediction and prediction_error["correct_label"] == label:
            return True
    return False

def add_prediction_error(prediction, label, example, prediction_errors):
    """
    If there is not already a prediction error with the given prediction and label, create one,
    initializing its examples list with the given example; otherwise update the existing prediction
    error's examples list with the given example.

    Arguments:
      prediction: the predicted label
      label: the correct label
      example:  the example to add to the prediction error's examples list
      prediction_errors: the list of prediction errors
    """
    # If there is not already a prediction error with the given prediction and label, create one
    if not prediction_error_exists(prediction, label, prediction_errors):
        prediction_errors.append({"predicted_label": prediction, "correct_label": label, "examples": [example], "count": 1})
    # Otherwise update the existing 
    else:
        for prediction_error in prediction_errors:
            if prediction_error["predicted_label"] == prediction and prediction_error["correct_label"] == label:
                if prediction_error["count"] < MAX_ERROR_EXAMPLES:
                    prediction_error["examples"].append(example)
                prediction_error["count"] += 1
                break

def score() -> dict:
    """ 
    Load augmented database from disk and iterate to compute the loss. 
    Display loss as a proportion.

    Returns dict with the following keys:
        n - number of rows in the dataset
        error_count - number of errors
        errors - list of dicts like 
          {"predicted_label": 1, "correct_label": 2, "count": 3, "examples": ["text1", "text2", "text3"]}
        responses - list dicts like
            {"response": "health", "count": 100}
        bad_predictions - number of predictions that were -1
    """
    # Prediction errors - array of dicts like 
    #   {"predicted_label": 1, "correct_label": 2, "count": 3, "examples": ["text1", "text2", "text3"]}
    # Counts are incident counts for the given predicted/correct label pair, examples is a rolling list of examples.
    prediction_errors = []
    response_counts = []

    ds = load_from_disk(AUGMENTED_DATASET_DIR)
    print("Read ", len(ds), " records from " + AUGMENTED_DATASET_DIR)
    print("First record: " , ds[0])
    
    # Iterate the dataset to compute error rate, fill prediction_errors and accumulate response counts
    error_count = 0
    bad_predictions = 0
    n = len(ds)
    for i in range(n):
        response = ds[i]["response"]
        found = False
        for response_count in response_counts:
            if response_count["response"] == response:
                response_count["count"] += 1
                found = True
                break
        if not found:
            response_counts.append({"response": response, "count": 1})
        correct = ds[i]["topic"]
        predicted = ds[i]["prediction"]
        if not correct == predicted and not predicted == -1:
            error_count += 1
            add_prediction_error(predicted, correct, ds[i]["text"], prediction_errors)
        if predicted == -1:
            bad_predictions += 1
    return {"n": n, "error_count": error_count, "errors": prediction_errors, "responses": response_counts, "bad_predictions": bad_predictions}

# Demo

make_predictions(prepare_dataset(), 1000)

results = score()
print("n: ", results["n"])
print("n incorrect: ", results["error_count"])
print("error rate: ", results["error_count"] / results["n"])
print("bad predictions: ", results["bad_predictions"])
print("Responses:", results["responses"])
sorted_errs = sorted(results["errors"], key=lambda x: x['count'], reverse=True)
print(len(sorted_errs), " distinct errors")
print("Top 10 errors:")
for i in range(0, 10):
    if i >= len(sorted_errs):
        break
    print("Error: ", "predicted:",yahoo_classes[sorted_errs[i]["predicted_label"]], " correct:",
           yahoo_classes[sorted_errs[i]["correct_label"]], " ", sorted_errs[i]["count"])
    for example in sorted_errs[i]["examples"]:
        print(example)
        print()

 
