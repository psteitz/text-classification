# yahoo questions categories
YAHOO_CLASSES= [
           "society or culture",
           "science or mathematics",
           "health",
           "education or reference",
           "computers or internet",
           "tokenizer",
           "business or finance",
           "entertainment or music",
           "family or relationships",
           "politics or government"
]

# NLI hyothesis templates
HYPOTHESIS_TYPE = 0
HYPOTHESIS_TEMPLATES = ["This example is {}.",
                        "This text is about {}.",
                        "This text comes under the heading {}.",
                        "The category of this text is {}.",
]
YAHOO_TEMPLATES = ["This question and answer are about {}.",
                   "This question and answer should be classified as {}.",
                   "The right category for this question and answer is '{}'.",
                   "The category of this question and answer is {}.",
                   "This question and answer relates to {}.",
                   "This question and answer is {}.",
                   "This question is {}.",
                   "This question relates to {}.",
                   "This question is about {}.",
                   "This question should be classified as {}.",
                   "The category of this question is {}.",
]
                     

# Database save locations
AUGMENTED_DATASET_DIR = "./hf_yahoo_data_augmented"

# Huggingface models
HF_MNLI = "facebook/bart-large-mnli"
HF_ZERO_SHOT = "zero-shot-classification"

# Default PRNG seed
DEFAULT_SEED = 42

# Huggingface datasets
HF_YAHOO_ANSWERS = 'yahoo_answers_topics'
