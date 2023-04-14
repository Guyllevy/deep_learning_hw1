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

The hyperparameters delta and lambda eem like two different hyperparameters, but in fact they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective.
the exact value of the margin between the scores arbitrary because the weights can shrink or stretch the differences arbitrarily. Hence, the only real tradeoff is how large we allow the weights to grow.

"""

part3_q2 = r"""
**Your answer:**

1. It seems like the linear model learns for each class a filter, which remembers the pixels which are the most likely to be light in the specific class. thus we get a filter that somewhat resembles a member of the class (e.g. the filter for the class 3 looks like the digit 3).

2. It is similar to KNN in that it quantifies proximity of examples.
the way it achieves that is different though:
KNN achives it by looking at the proximity of the test sample to all trained samples (by euclidian distance - lower is good).
and the linear classifier does that by calculating a proximity to a learned filter (by dot product - higher is good).

"""

part3_q3 = r"""
**Your answer:**

1. I would say the learning rate is just about right (good).
The graph goes down quickly enough and reaches some saturation / limit.
If the learning rate would have been too low, we wouldn't reach saturation in the same number of epochs.
If the learning rate would have been too high, the graph wouldn't have been soo smooth and monotone.

2. I would say it is slightly overfitted though it is hard to tell from the graph.
the validation accuracy graph reaches saturation and maybe slightly decreases at the end (while the train accuracy still goes slightly up), which is a sign for overfitting.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see is low residuals, with no noticable pattern to the residuals. a noticable pattern will indicate that the model missed some pattern it could have learned from.

compared to the plot we got with 5 features, the last plot that we got with CV has lower residuals which indicates better fitness of the model.

"""

part4_q2 = r"""
**Your answer:**

1. Depends on what you call linear - the model is still linear on the new nonelinear features, but is not linear on the original features. so we could say overall it is not linear.
2. We cant fit ANY function, for example we cant fit a random function with nothing to learn from previous data.
3. The decision boundery will no longer be a hyperplane, we can think of a simple classifier that returns 1 if x**2 > 5 and 0 otherwise, with feature x and added nonlinear feature x**2, we can fit such model perfectly with a linear model on x**2, and the decsion boundery would be parabolic.

"""

part4_q3 = r"""
**Your answer:**

1. The advantage of using logspace for CV is that it allows us to search a wide range of values for lambda while avoiding a bias towards smaller or larger values. This is because the impact of lambda on the model's performance is often non-linear, and using a logarithmic scale helps to evenly sample values across this non-linear range.

2. In cross validation the total amount of times we fit a model to data is K * H where K is the number of folds and H is the number of possible combinations of hyperparameter values. in our example it is k_folds * len(degree_range) * len(lambda_range).

"""

# ==============
