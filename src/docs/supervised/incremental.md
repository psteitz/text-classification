# Incremental training

## Overview
Incremental training breaks a training set into blocks and uses a form of teacher-student training to train a sequence of models. Student models learn natural label probability distributions from their teachers and training data.  Training examples that teachers get right are used to move student models toward natural distributions. Instead of using one-hot vectors as training targets, the student trainer uses the teacher's estimates (when these are correct and stable) in cross-entropy loss calculation. All student model logits are taken into account in the loss function when the parent's model responses are used.

We assume that the texts fall into natural distributions vis a vis the labels and in many cases, the correct distribution for an input text is not a point mass on one label.  The intuition here is that multiple labels may apply naturally to some input texts and the correct one identified by a labeler is just the one with the biggest mass. Our hypothesis is that if we can get student models to learn natural distributions by targeting teacher models'correct estimates of them rather than one-hot vectors concentrated on the correct label, we can train models that will generalize better.

### Alogorithm
Start with a full training set $T$ with $n$ elements. Decide on the number of training blocks, $k$, to form from $T$.  If the base model is pretrained, it doesn't need to consume a block, so the number of blocks is one less than the number of student models to train. Let $B_0, ..., B_{k-1}$ be the sample blocks.

Either set $M_0$ to a pretrained model or train $M_0$ by fine-tuning a base transformer model using the data from $B_0$.

Let $l$ be the number of student models. For each $i = 0,...l-1$, train student model $M_{i+1}$ using $M_{i}$ with data from sample block $B_i$[^1].

Instead of always using one-hot target vectors in cross-entropy loss calculation in the student trainer's loss computation, use the teacher model's output distribution if the teacher model's output is correct and stable. Here <i>stable</i> means that the model's mass on the correct answer is beyond a configured stability threshold over the next highest value. So for example, if the stability threshold is $.2$, the correct label index is $1$ and the model (after softmax) returns $[.2, .6, .2]$, that is a correct and stable model response, but $[.3, .4, .3]$ is correct but not stable.


## Implementation
Let $M_{0}$ be either a pre-trained (but not fine-tuned) transforomer model or a pre-trained nli model.

If $M_0$ is an nli model, when creating $M_1$, we use a base transformer model as the student. Thereafter, we train $M_{i}$ by fine-tuning $M_{i - 1}$ using data from sambple block $B_{i}$[^1].

For each sample $s$ from $B_i$, if $M_{i - 1}$ responds with a correct and stable answer, we use the model's softmax response as the target in cross entropy loss rather than a one-hot vector; otherwise we use the standard loss function when evaluating the model's output on $s$. 

## Implementation using HuggingFace transformers and Trainer
For student training, we need to replace the standard cross-entrooy loss function that uses a one-hot target vector with one that can take a distribution, computed using the teacher model.  The easiset way to do this is to subclass the [HuggingFace Trainer](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#trainer) class, overriding its [compute_loss](https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L2791) function.  The implementation code is in the [supervised](https://github.com/psteitz/text-classification/blob/main/src/main/python/supervised) module, which includes the [subclassed HuggingFace Trainer](https://github.com/psteitz/text-classification/blob/main/src/main/python/supervised/StudentTrainer.py)

[^1] If $M_0$ consumes $B_0$ then $M_{i}$ is trained using data from $B_{i + 1}$ thereafter.
