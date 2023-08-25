"""
Zero-shot classification example using llama-2-70b-chat accessed via the Replicate API

Requires pip install replicate
Export Replicate API key as REPLICATE_API_TOKEN.
To get a key:  https://replicate.com.
Free usage is limited.
"""

from datasets import load_dataset, Dataset
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

replicate_model_name = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"

categories_string = ""
for item in yahoo_classes:
    categories_string += item + "\n"

system_prompt = "Your job is to find the right category for each question and answer pair provided.\n \
The categories are: \n" + categories_string
                
def classify(question_anawer : str) -> str:
   """
    Classify the given question/answer pair.
   """
   model_response = replicate.run(
        replicate_model_name,
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
        text - the text to search for in yahoo_classes
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
    """
    dataset = load_dataset('yahoo_answers_topics', split='test')
    new_column = ["null"] * len(dataset)
    dataset = dataset.add_column("text", new_column)
    return dataset.map(fill_text).remove_columns(["question_title", "question_content", "best_answer"])

def add_prediction(dataset, predictions, i):
    """
    Add the prediction for the given record to the predictions column.
    """
    text = dataset[i]["text"]
    predictions[i] = yahoo_index_from_text(classify(text))
    print("Prediction: ", predictions[i])
    print("Text: ", text)

def make_predictions(dataset: Dataset, max_records : int = -1):
    """
    Make predictions for the given dataset and save the dataset, augmented with predictions.

    Arguments:
        dataset - the dataset to augment with predictions based on "text" column
        max_records - if this is positive, don't process the entire dataset, just sample max_records records
    """
    if max_records > 0:
        dataset = dataset.select(range(0, max_records))
    predictions = [-1] * len(dataset)
    num_predictions = 0
    for i in range(0, len(dataset)):
        add_prediction(dataset, predictions, i)
        num_predictions += 1
        if num_predictions % 5 == 0:
            print("Predicted ", num_predictions, " records.")
            print("Saving augmented dataset to disk.")
            # Drop the previous predictions column if it exists
            if "prediction" in dataset.column_names:
                dataset = dataset.remove_columns(["prediction"])
            # Add copy of current predictions column
            dataset = dataset.add_column("prediction", predictions.copy())
            # Save to disk
            dataset.save_to_disk("yahoo_answers_topics_augmented")
            print("Saved augmented dataset to disk.")
    print("Final save of augmented dataset to disk.")
    dataset.save_to_disk("yahoo_answers_topics_augmented")

make_predictions(prepare_dataset(), 100)
"""
print("System prompt: " + system_prompt)
print("Prompt: What elements make up water? Oygen and hydrogen.")
show_response(classify("What elements make up water? Oygen and hydrogen."))
"""
