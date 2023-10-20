"""
Zero-shot and few-shot classification using llama-2-70b-chat accessed via the Replicate API

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

# Classification labels
LABELS = [
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

# Number of examples to hold for each prediction error
MAX_ERROR_EXAMPLES = 5

LABEL_LIST = ""
for item in LABELS:
    LABEL_LIST += item + "\n"

SYSTEM_PROMPT_ZERO_SHOT = "Your job is to find the right category for each question and answer pair provided.\n\
Please do not repeat the question in your response or add any explanation for your answer.\n\
Just respond with the best category name.\nFor example: 'family or relationships'.\n\
Please respond with the category name only.  Do not include any other text in your response and make sure that your response \
matches one of the categories exactly."

SYSTEM_PROMPT_FEW_SHOT = "Your job is to find the right category for each question and answer pair provided.\n\
You will be provided with a few examples of questions and answers and their categories.\n\
You will be asked to classify a question and answer pair.\n\
Please do not repeat the question in your response or add any explanation for your answer.\n\
Just respond with the best category name.\n For example: 'family or relationships'.\n\
Please respond with the category name only. Do not include any other text in your response and make sure that your response \
matches one of the categories exactly.\n\
The categories are: \n" + LABEL_LIST + "\n"

                
def classify_zero_shot(question_anawer : str) -> str:
   """
    Classify the given question/answer pair.

    Arguments:
        question_anawer - the question/answer pair to classify
    Returns:
        the model response as a string
   """
   model_response = replicate.run(
        REPLICATE_MODEL_NAME,
        input={"system_prompt": SYSTEM_PROMPT_ZERO_SHOT, "prompt": question_anawer}
    )
   # Consume the model response stream and build the output string from it.
   output = ""
   for item in model_response:
        output += item
   return output

def classify_few_shot(question_answer : str, examples : List) -> str:
    """
    Classify the given question/answer pair using a few-shot prompt.

    Arguments:
        question_answer - the question/answer pair to classify

        examples - the examples to use for the few-shot prompt

    Returns:
        the model response as a string
    """
    # Build the prompt
    prompt = ""
    # Load the examples
    for example in examples:
        prompt += "text: " + example["text"] + "\n"
        prompt += "label: " + LABELS[example["label"]] + "\n\n"
    # Ask for the label for the question/answer pair
    prompt += "text: " + question_answer + "\nClabel: "
    print("Prompt: ", prompt)
    print("text: ", question_answer)
    print("Correct label: ", LABELS[examples[0]["label"]] + "\n")
    # Run the model
    model_response = replicate.run(
        REPLICATE_MODEL_NAME,
        input={"system_prompt": SYSTEM_PROMPT_FEW_SHOT, "prompt": prompt}
    )

     # Consume the model response stream and build the output string from it.
    output = ""
    for item in model_response:
        output += item
    print("Model response: ", output)
    return output

def label_index_from_text(text: str) -> int:
    """
    Return the index of text in yahoo_classes.

    Returns the index of the first class in yahoo_classes
    that is a substring of text.lower(); or -1 if there is no such class.

    LLM model responses may include extra garbage text,
    so we search for the label in the response.

    Arguments:
        text - input text
    Returns:
        the index of the first class in yahoo_classes that is a substring of text.lower();
       -1 if there is no such class.
    """
    lower = text.lower()

    for i in range(0, len(LABELS)):
        if LABELS[i] in lower:
            return i 
    print("Error: yahoo class not found: " + text)
    return -1


def fill_text(rec : Dict):
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

def add_zero_shot_prediction(dataset : Dataset, predictions : List, responses : List, i : int):
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
    # Make prediction using LLM model
    model_response = classify_zero_shot(text)
    responses[i] = model_response
    # model_response may or may not match a class in yahoo_classes exactly.
    # Set prediction[i] to the first index in yahoo_classes that embeds model_response or -1 if none does.
    predictions[i] = label_index_from_text(model_response)

    print("Prediction: ", predictions[i])
    print("Text: ", text)

ZERO_SHOT = 0 # Use zero-shot prompt
FEW_SHOT_SUPERVISED = 1 # Create a few-shot prompt, based on examples selected based on a supervised model's confidence
FEW_SHOT_RANDOM = 2 # Create a few-shot prompt, based on randomly selected examples
# Load SUPERVISED_EXAMPLES and RANDOM_EXAMPLES by sampling from the augmented dataset
def sample_random(dataset : Dataset, records_per_class : int = 1) -> List:
    """
    Return a list of example records from the given dataset.

    For each class in yahoo_classes, the output list includes up to records_per_class examples of the class.

    If there are more than records_per_class examples of the class, select records_per_class randomly.

    Arguments:
        dataset - the dataset to sample
        records_per_class - the number of examples to include for each class
    Returns:
        a list of example records from the given dataset including up to records_per_class examples of each class
    """
    indices = []
    for i in range(0, len(LABELS)):
        # Get the indices of all records where prediction == i
        class_indices = dataset.filter(lambda rec: rec["prediction"] == i)
        # Select up to records_per_class indices from among class_indices randomly
        class_indices = random.sample(range(0, len(class_indices)), records_per_class)
        indices += class_indices
    sample_dataset = dataset.select(indices)
    # Convert sample_dataset to a list of records
    out = []
    for i in range(0, len(sample_dataset)):
        out.append(sample_dataset[i])
    return out

def sample_supervised(dataset : Dataset, records_per_class : int = 1) -> List:
    """
    Return a list of example records from the given dataset. 

    For each class in LABELS, the list includes up to records_per_class examples of that class.

    If the number of examples of the class is less than records_per_class, the list includes all examples of the class;
    otherwise records_per_class examples are selected based on model confidence as measured by softmax score.

    The Dataset has to have "prediction" and "score" columns added by running a base datset through a supervised model.
    The prediction is an int index in to LABELS and the score is the model softmax score for the prediction.

    Arguments:
        dataset - the dataset to sample from - needs to be an augmented dataset with "prediction" and "score" columns
        records_per_class - the number of examples to include for each class
    Returns:
        a list of example records from the given dataset including up to records_per_class examples of each class
    """
    indices = []
    for i in range(0, len(LABELS)):
        # Get the indices of all records where prediction == i, sort by max score,
        # and select the first records_per_class
        class_indices = dataset.filter(lambda rec: rec["prediction"] == i)\
        .sort("score", reverse=True)\
        .select(range(0, records_per_class))
        indices += class_indices
    return dataset.select(indices)

RANDOM_SELECTION_STRATEGY = 0
SUPERVISED_SELECTION_STRATEGY = 1
RECORDS_PER_CLASS = 1
def sample(dataset : Dataset, records_per_class : int = RECORDS_PER_CLASS, strategy : int = RANDOM_SELECTION_STRATEGY) -> List:
    """
    Return a list of example records from the given dataset.

    For each class in yahoo_classes, the list includes records_per_class examples of that class.

    If strategy is 0 (random), the examples are selected randomly from the dataset.

    If strategy is 1 (supervised), examples are selected by running the dataset through a supervised model
    and selecting examples based on model confidence as measured by softmax score.

    """
    if strategy == RANDOM_SELECTION_STRATEGY:
        return sample_random(dataset, records_per_class)
    elif strategy == SUPERVISED_SELECTION_STRATEGY:
        return sample_supervised(dataset, records_per_class)
    else:
        print("Error: sample: invalid strategy: ", strategy)
        return []

# Load the augmented dataset that is expected to contain "prediction" and "score" columns filled
# by running a base dataset through a supervised model.
AUGMENTED_DATASET = load_from_disk(AUGMENTED_DATASET_DIR)
# Pull samples of examples to use in creating few-shot prompts
SUPERVISED_EXAMPLES = sample(AUGMENTED_DATASET, RECORDS_PER_CLASS, SUPERVISED_SELECTION_STRATEGY)
RANDOM_EXAMPLES = sample(AUGMENTED_DATASET, RECORDS_PER_CLASS, RANDOM_SELECTION_STRATEGY)

def add_few_shot_prediction(dataset : Dataset, predictions : List, responses : List, i : int, prompt_type : int = FEW_SHOT_RANDOM):
    """
    Make prediction based on "text" field and store prediction and raw model response 
    for the given record to the predictions and responses columns, resp.
    Use a few-shot prompt, based on examples selected using a supervised model (FEW_SHOT_SUPERVISED)
    or randomly (FEW_SHOT_RANDOM), based on the value of prompt_type.

    Arguments:
        dataset - the dataset 
        predictions - the predictions column
        responses - the responses column
        i - the index of the record to process
        prompt_type - the type of few shot prompt to use for the model
    """
    text = dataset[i]["text"]
    # Make prediction using LLM model, with a prompt build either from supervised or random examples
    if prompt_type == FEW_SHOT_SUPERVISED:
        model_response = classify_few_shot(text, SUPERVISED_EXAMPLES)
    else:SUPERVISED_EXAMPLES
        model_response = classify_few_shot(text, RANDOM_EXAMPLES)
    
    # add model response to responses column
    responses[i] = model_response
    # model_response may or may not match a class in LABELS exactly.
    # Set prediction[i] to the first index in LABELS that embeds model_response or -1 if none does.
    predictions[i] = label_index_from_text(model_response)

    print("Prediction: ", predictions[i])
    print("Text: ", text)


def make_predictions(dataset: Dataset, max_records : int = -1, predictions_save_interval : int = 5, prompt_type : int = ZERO_SHOT):
    """
    Make predictions for the given dataset and save the dataset,
    augmented with predictions and model response text.
     
    If max_records is positive, select a random sample of max_records to process.

    Save the augmented dataset to disk every predictions_save_interval records.

    Arguments:
        dataset - the dataset to augment with predictions based on "text" column

        max_records - if this is positive and smaller than len(dataset), 
                      sample max_records records from dataset randomly
                      DEFAULT: unlimited

        predictions_save_interval - save the augmented dataset to disk after every predictions_save_interval records
                                    DEFAULT: 5

        prompt_type - the type of prompt to use for the model   
                      DEFAULT: ZERO_SHOT

    """
    if max_records > 0 and max_records < len(dataset):
        indices = random.sample(range(0, len(dataset)), max_records)
        dataset = dataset.select(indices);
    predictions = [-1] * len(dataset)
    responses = ["null"] * len(dataset)
    num_predictions = 0
    for i in range(0, len(dataset)):
        if prompt_type == ZERO_SHOT:
            add_zero_shot_prediction(dataset, predictions, responses, i)
        else:
            add_few_shot_prediction(dataset, predictions, responses, i, prompt_type)
        num_predictions += 1
        if num_predictions % predictions_save_interval == 0:
            print("Predicted ", num_predictions, " records.")
            print("Saving augmented dataset to disk.")

            # Replace the dataset predictions and responses columns with copies
            # of predictions and responses.
            
            # Drop the database columns if they exist
            if "prediction" in dataset.column_names:
                dataset = dataset.remove_columns(["prediction"])
            if "response" in dataset.column_names:
                dataset = dataset.remove_columns(["response"])

            # Recreate the database columns using copies of predictions and responses
            dataset = dataset.add_column("llm_prediction", predictions.copy())
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

make_predictions(prepare_dataset(), 10, 5, FEW_SHOT_SUPERVISED)

 
