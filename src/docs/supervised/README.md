# Incremental training

Incremental training breaks a training set into blocks and uses a form of teacher-student training and a modified loss calculation to train a sequence of models.

The method works as follows.

Start with full training set $T$ with $n$ elements. Decide on the number of training blocks, $k$ = the number of student models to train.
Train $M_0$ from $S_0 = n/k$ elements randomly selected from T.  Train $M_0$ by fine-tuning a base transformer model in the normal way, which computes cross-entropy loss using one-hot vectors with "1" in the spot for the correct label.  Then train student models $M_1, ..., M_k$ using sample blocks, giving them teacher output, transforming as needed to get the probability for the correct label above a configured threshold.  The intuition here is that the "correct" model output is a distribution and that distribution may have a decent amount of mass not on the correct answer.  As students get better, they learn to map inputs to the natural distributions.

Create training set $T_1$ as follows:
Select $S_1 = n/5$ random elements from $T - S_1$. For each $s$ in $S_1$,
Let $v_s$ be the full softmax array returned by $M_0$ when provided with s["text"] as text input.
Instead of computing the cross-entropy loss using a one-hot array that concentrates all of the mass on the correct label, look at the softmax vector and make proportional reductions in non-correct dimensions and increases in the correct one until the correct one is higher than p where p is a configurable threshold (.9 by default).  Optionally, set a threshold below wich a one-hot vector is used.

Create $T_2$ similarly, but now using $M_1$ to score $S_2$, another $n/k$ elements from $T - (S_0 \cup S_1)$.

Repeat until all of $T$ is used and return the final model.

## Implementation using HuggingFace transformers and Trainer
For student training, we need to replace the standard cross-entrooy loss function that uses a one-hot target vector with one that can take a distribution, computed useing the teacher model.  The easiset way to do this is to subclass the [HuggingFace Trainer](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#trainer) class, overriding its [compute_loss](https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/trainer.py#L2791) function.
