r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

Statement 1 : "The test set allows us to estimate our in-sample error."
This statement is false, the in-sample error is the error we get on the examples from the training set.
The test set allows us to estimate the out-sample error, the error we get on example we haven't seen while training the model.
So we can get small error on the training set while the error on the test set is high, and so the test set is not a good estimator to the in-sample error.

Statement 2 : "Any split of the data into two disjoint subsets would constitute an equally useful train-test split."
This statement false, for example a split which has 1% of the examples in the training set and the rest in the test set will probably won't do very well compared to another split of say 50-50. Thats because we learned very little from existing examples.
Or another example: if we use a split in which there is small amount of one class samples compared to the other class samples, the split is imbalanced and therefore could lead to poor model performance.

Statement 3 : "The test-set should not be used during cross-validation."
This is true, the purpose of the test set is to evaluate the final model's ability to generelize. so if we use the test set during cross validation we learn from it which loses this purpose.

Statement 4 : "After performing cross-validation, we use the validation-set performance of each fold as a proxy for the model's generalization error."
This is true, after performing cross-validation, we use the average of the validation set performance across all folds to estimate the model's generalization error.

"""

part1_q2 = r"""
**Your answer:**

The friends approach is not justified as he chose his model based on the test results.
By doing this he effectivly learned from the test set.
That is a wrong approach because the test set is suppose to evaluate a model's ability to generelize over the training data.



"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing k leads to improved generalization only up to a point,
when k is large the algorithm, KNN when predicting a test example will take in to account examples which are far away from the test example, and so are less informative (less likely to be the same class as the test example).
For that reason we will get worse results when k is increased from some point.

"""

part2_q2 = r"""
**Your answer:**

Explain why (i.e. in what sense) using k-fold CV, as detailed above, is better than:

Training on the entire train-set with various models and selecting the best model with respect to train-set accuracy.
Training on the entire train-set with various models and selecting the best model with respect to test-set accuracy.

Better than 1: Because training on the entire train-set can cause overfitting the examples of the train-set. A model can do very well on the train-set but with very poor generalization ability. k-fold CV solves this by extimating each models generalization ability.

Better than 2: Because choosing a model based on performance on the test set means we learned a model using the test set.
The test set is meant for evaluating the models generalization ability, and by learning from it we miss this goal.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
