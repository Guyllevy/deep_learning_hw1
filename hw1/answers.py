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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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
