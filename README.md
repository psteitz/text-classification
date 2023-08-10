# text-classification
This repo contains examples of three different ways to do text classification. The examples use open source data. Repeating these experiments with other datasets would be helpful. PRs reporting results ideally including datasets are most welcome,

## Supervised transformer-based
Start with a pre-trained transformer model, add layer to classify embeddings of input texts anf fine-tune with labelled data.
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

### zero-shot mnli-trained classifier
Translate the classfication problem into an inference problem

Look at entailment probabilities for 
* __hypothesis__ "This text is about {}" and 
* __premise__ text to be classified with {} filled in for each of the labels

The mnli model gives us three probabilities for each classufication label against each of the input texts:
* probability of entailment - the probability that the premise entails the hypothesis
* neutral probability - the probability that the premise and hypothesis are compatible, but not related
* probability of contradiction - the probability that the premise contradicts the hypothesis.

Zero-shot, mnli-based classification only looks at the first set of probabilities, returning the classification with the highest entailment probability.

Because you are asking the model to semantically relate the classification with the text, it makes a big difference what text labels you use to express the classifications.  If your classification labels are clear and don't overlap that much, things work better.
These models are slower than supervised classifiers, but faster than LLMs.

## Zero-shot small model, inferernce-trained, fine-tuned
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


## Few shot, LLM