#Incremental training

Incremental training breaks a training set into blocks and uses a form of teacher-student training and a modified loss calculation to train a sequence of models.

The method works as follows.

Start with full training set T with n elements.
Train m_0 from s_0 = n/5 elements randomly selected from T.  Train m_0 starting with a base transformer model and fine-tuning it in the normal way, which computes cross-entropy loss using one-hot vectors with "1" in the spot for the correct label.  Then train student models m_1, ... m_k using sample blocks, giving them teacher output, transforming as needed to get the probability for the correct label above a configured threshold.  The intuition here is that the "correct" model output is a distribution and that distribution may have a decent amount of mass not on the correct answer.  As students get better, they learn to correctly generate these distributions.

Create training set t_1 as follows:
Select s_1 = n/5 random elements from T - s_1. For each s in s_1,
Let v_s be the full softmax array returned by m_0 when provided with s["text"] as text input.
Instead of computing the cross-entropy loss using a one-hot array that concentrates all of the mass on the correct label, look at the softmax vector and make proportional reductions in non-correct dimensions and increases in the correct one until the correct one is higher than p where p is a configurable threshold (.9 by default).  Optionally set a threshold below wich a one-hot vector is used.

Create t_2 similarly, but now using m_1 to score s_2, another n/5 elements from T - (s_0 U s_1).

Repeat 3 more times, using all of T and return the final model.