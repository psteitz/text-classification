from ast import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Set labels to yahoo questions categories
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
                      model="facebook/bart-large-mnli")
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


sequence = "What elements make up water? Oygen and hydrogen."
print("HuggingFace classifier: ", huggingface_zero_shot_classify(sequence, yahoo_classes))

print("\nRaw NLI logits");
for label in yahoo_classes:
    hypothesis = f'This example is {label}.'
    print(f'Premise: {sequence}')
    print(f'Hypothesis: {hypothesis}')
    print(f'Raw NLI logits: {raw_nli(sequence, hypothesis)}')
    print(f'HuggingFace zero-shot classification: {huggingface_zero_shot_classify(sequence, yahoo_classes)}')
    print()

print("Output from nli model");
hypothesis = 'This example is politics or government.'
print(f'Premise: {sequence}')
x = tokenizer.encode(sequence, hypothesis, return_tensors='pt',
                        truncation=True)
print(nli_model(x.to(device)))


