# text-classification
This repo contains examples of three different ways to do text classification.
1. Supervised: fine-tuned transformer model
2. Zero shot nli: pre-trained transformer nli model
4. Few shot: chat-trained LLM

The examples use open source modles and data. Huggingface datasets are used to manage training data and store model prediction results. Huggingface models create predictions.

Repeating these experiments with other datasets would be helpful. PRs reporting results ideally including datasets are most welcome,


## Supervised transformer-based
Start with a pre-trained transformer model, add a layer to classify input texts anf fine-tune with labelled data.
See [HuggingFace tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification).
* Usually most accurate 
* Always the fastest
* Always lowest cost
* Models can be small

## Zero-shot using mnli model
Start with a pre-trained transformer model fine-tuned for mnli tasks and translate the classification task into an inference task.
* Good for cold start, where labeled data can't be acquired or when classifications are dynamic. 
* Fair to good accuracy, depending on how clear the classification labels are and how much they overlap
* Slower than supervised, faster than LLM
* Models can be small

### Zero-shot mnli-trained classifier
Translate the classfication problem into an inference problem

Look at entailment probabilities for 
* __hypothesis__ "This text is about {}" and 
* __premise__ text to be classified

The mnli model gives us three probabilities for each classufication label against each of the input texts:
* probability of entailment - the probability that the premise entails the hypothesis
* neutral probability - the probability that the premise and hypothesis are compatible, but not related
* probability of contradiction - the probability that the premise contradicts the hypothesis.

Zero-shot, mnli-based classification only looks at the first set of probabilities, returning the classification with the highest entailment probability.

Because you are asking the model to semantically relate the classification with the text, it makes a big difference what text labels you use to express the classifications.  If your classification labels are clear and don't overlap that much, things work better.
These models are slower than supervised classifiers, but faster than LLMs.

### Zero-shot small model, inferernce-trained, fine-tuned
You can also fine-tune a Zero-shot, mnli classification model.

__Caution__ Once you have a bunch of labelled, corrected data it is generally best to move to a supervised model.

Fine-tuning an mnli classifier means fine-tuning the mnli model's inference task.  Training records for this task 
have premise, hypothesis, and label keys.
* premise is the text to classify.
* hypothesis is a statement like "This text is about <class>." where <class> is the classification text 
* label is 0, 1 or 2 where 0 means contradiction, 1 means neutral and 2 means entailment
Classification examples can be used as fine-tuning input data, but they need to be augmented with training records that are neutral and / or contradiction examples. The training data has to be at least a little bit balanced across the labels. 

Paper recommendation:  make random (uniform) choice between neutral and contradicts for one randomly selected other classification.
Alternatives:
* Make the choice based on what the current model says
* Add more records for other labels based on what current model says


## Chat-trained LLM
Use a chat-trained LLM to classify input texts.  Two ways to do this:
1. Zero-shot - system prompt provides instructions including labels and prompt provides the text to be classified
2. Few-shot - prompt provides "canonical" examples of classifications and then asks the LLM to classify the input similarly
