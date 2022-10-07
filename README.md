# iNeuronMLQuestions

This repository contains important Machine Learning interview questions.

## Fundamentals of Machine Learning
Q1. What is machine learning?

Q2. Why do we use machine learning?

Q3. How many types of machine learning systems do we have?

Q4. What is the difference between supervised and unsupervised learning system?

Q5. What is semi-supervised learning system?

Q6. What is reinforcement learning?

Q7. What is batch and online learning?

Q8. What is the difference between instance-based and model-based learning system?

Q9. What are the challenges faced by the machine learning systems?

Q10. What is out-of-core learning?

Q11. If a company wants to develop a system which can be used if a mail is ham or spam mail, then which type of learning system should be used?

Q12. If a company wants to predict a model-based learning system which can be used to predic the cost of a car based on certain parameters, then which type of learning system should be used?

Q13. If a company wants to train a robot which has the ability to take decisions based on the current state of the robot, then which type of learning system should be used?

## Machine learning Project lifecycle

Q14. Describe the lifecycle of the machine learning system.

Q15. What is the necessity of understanding the problem statement and creating a well defined architecture?

Q16. Why do we create a seprate workspace for every problem?

Q17. What are the different sources which can be used as a source of data gathering?

Q18. What is the data annotation?

Q19. What are the different steps involved in data wrangling?

Q20. What are the steps involved in model development?

Q21. What are the different steps involved in model training?

Q22. What is hyperparameter tuning?

Q22. What are the different steps involved in model evaluation?

Q23. What is model over fitting and under fitting?

Q24. What is model deployment?

Q25. What are the different sources where we can deploy our model?

Q26. What is model monitoring and how can we do it?

Q27. What is model retraining?

Q28. What are the conditions when we need to do model retraining?

Q29. Download the [housing data](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and train a machine learning model and can be used to predict the price of house given the required parameters.

Q30. Try creating a single pipeline that does every steps from data preparation to model prediction.

## Linear Regression Algorithms
Q31. What is meant by regression?

Q32. What is meant by linear regression?

Q33. Write a python function that take an input value, weight and bais as an input and returns the returns the prediction.

Q34. What is normal equation?

Q35 What is an error?

Q36. What is the difference between error and cost function?

Q37. What is gradient descent?

Q38. What is random initialization in gradient descent?

Q39. What is the role of learning rate?

Q40. What will happen if the learning rate is too less?

Q41 What will happen if the learning rate is too large?

Q42. What is local and global minima?

Q43. What is batch gradient descent?

Q44. What is convergence rate?

Q45. What is stochastic gradient descent?

Q46. What is mini batch gradient descent?

Q47. Explain polymonial regression mathematically.

Q48. How learning curves avoids underfitting and overfitting problems?

Q49. What is regularization?

Q50. How ridge regression helps reduce the overfitting problem?

Q51. How lasso regression helps reduce the overfitting problem?

Q52. What is the difference between ridge and lasso regression?

Q53. Explain Elastic Net.

Q54. How can we use early stopping to reduce overfitting?

Q55. How is logistic regression used for classification?

Q56. Explain the use of softmax function in logistic regression.

Q57. Can we use logistic regression for multiclass classification?

Q58. What is a decision boundary?

Q59. Explain softmax regression.

Q60. What is cross entropy?

Q61. Which Linear Regression training algorithm can you use if you have a training
set with millions of features?

Q62. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?

Q63. Can Gradient Descent get stuck in a local minimum when training a Logistic
Regression model?

Q64. Do all Gradient Descent algorithms lead to the same model, provided you let
them run long enough?

Q65. Suppose you use Batch Gradient Descent and you plot the validation error at
every epoch. If you notice that the validation error consistently goes up, what is
likely going on? How can you fix this?

Q66. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?

Q67. Which Gradient Descent algorithm (among those we discussed) will reach the
vicinity of the optimal solution the fastest? Which will actually converge? How
can you make the others converge as well?

Q68. Suppose you are using Polynomial Regression. You plot the learning curves and
you notice that there is a large gap between the training error and the validation
error. What is happening? What are three ways to solve this?

Q69. Suppose you are using Ridge Regression and you notice that the training error
and the validation error are almost equal and fairly high. Would you say that the
model suffers from high bias or high variance? Should you increase the regulari‐
zation hyperparameter α or reduce it?

Q70. Implement Batch Gradient Descent with early stopping for Softmax Regression
(without using Scikit-Learn).

## SVM
Q71. Explain linear SVM classification.

Q72. What is large margin classification?

Q73. What is a support vector?

Q74. Are SVMs sensitive to scale of data?

Q75. What is hard margin classification? What is it's drawback?

Q76. What is soft margin classification?

Q77. What is a kernel in SVM?

Q78. How can we perform classification using SVC on huge datasets or streaming datasets?

Q79. Explain non-linear SVM classification.

Q80. What is polymonial kernel and how can we use it for non-linear SVM classification?

Q81. Is there any limitation of polymonial kernel?

Q82. What is kernel trick?

Q83. How can we use similarity features to tackle nonliner problem?

Q84. What is Gaussian RBF kernel?

Q85. Can we use SVMs for regression? If yes, explain how.

Q86. How SVM uses decision function to make predictions?

Q87. What is Dual Problem?

Q88. Explain Kernelized SVM.

Q89. What are online SVMs?

Q90. Why is it important to scale the inputs when using SVMs?

Q91. Can an SVM classifier output a confidence score when it classifies an instance?

Q92. Should you use the primal or the dual form of the SVM problem to train a model
on a training set with millions of instances and hundreds of features?

Q93. Say you’ve trained an SVM classifier with an RBF kernel, but it seems to underfit
the training set. Should you increase or decrease γ (gamma)? What about C?

Q94. How should you set the QP parameters (H, f, A, and b) to solve the soft margin
linear SVM classifier problem using an off-the-shelf QP solver?

Q95. Train a LinearSVC on a linearly separable dataset. Then train an SVC and a
SGDClassifier on the [iris dataset](https://www.kaggle.com/datasets/uciml/iris). See if you can get them to produce roughly the same model.

Q96. Train an SVM classifier on the MNIST dataset. Since SVM classifiers are binary
classifiers, you will need to use one-versus-the-rest to classify all 10 digits. You
may want to tune the hyperparameters using small validation sets to speed up the
process. What accuracy can you reach?

Q97. Train an SVM regressor on the [housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).


## Decision Trees
Q98. What are black box and white box models?

Q99. Explain the working of decision trees.

Q100. Can you import the decision trees in a graphical format for visualization?

Q101. Can we train a decision tree wihout scaling the data?

Q102. What is entropy? What is the mathematical formula for it?

Q103. What is information gain? How is mathematically calculated?

Q104. What is ginni impurity and how is it calculated?

Q105. What is CART(Classification and Regression Tree) algorithm?

Q106. What is the cost fuction of CART for classification?

Q107. Why is CART a greedy algorithm?

Q108. What is parametric model?

Q109. What is non parametric model?

Q110. What al regularization parameters can you use in decision tree?

Q111. What is pruning?

Q112. When do we need to perform pruning?

Q113. Expalin pre pruning.

Q114. What is post pruning?

Q115. What is the difference between decision tree classifier and decision tree regressor?

Q116. What is the cost fuction of CART for regression?

Q117. List some instabilities of decision tree.

Q118. What is the approximate depth of a Decision Tree trained (without restrictions)
on a training set with one million instances?

Q119. Is a node’s Gini impurity generally lower or greater than its parent’s? Is it gener‐
ally lower/greater, or always lower/greater?

Q120. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing
max_depth?

Q121. If a Decision Tree is underfitting the training set, is it a good idea to try scaling
the input features?

Q122. If it takes one hour to train a Decision Tree on a training set containing 1 million
instances, roughly how much time will it take to train another Decision Tree on a
training set containing 10 million instances?

Q123. If your training set contains 100,000 instances, will setting presort=True speed
up training?

Q124. Train and fine-tune a Decision Tree for the moons dataset by following these
steps:

    a. Use make_moons(n_samples=10000, noise=0.4) to generate a moons dataset.

    b. Use train_test_split() to split the dataset into a training set and a test set.

    c. Use grid search with cross-validation (with the help of the GridSearchCV
    class) to find good hyperparameter values for a DecisionTreeClassifier.
    Hint: try various values for max_leaf_nodes.

    d. Train it on the full training set using these hyperparameters, and measure
    your model’s performance on the test set. You should get roughly 85% to 87%
    accuracy.

Q125. Grow a forest by following these steps:

    a. Continuing the previous exercise, generate 1,000 subsets of the training set,
    each containing 100 instances selected randomly. Hint: you can use ScikitLearn’s ShuffleSplit class for this.

    b. Train one Decision Tree on each subset, using the best hyperparameter values
    found in the previous exercise. Evaluate these 1,000 Decision Trees on the test
    set. Since they were trained on smaller sets, these Decision Trees will likely
    perform worse than the first Decision Tree, achieving only about 80%
    accuracy.

    c. Now comes the magic. For each test set instance, generate the predictions of
    the 1,000 Decision Trees, and keep only the most frequent prediction (you can
    use SciPy’s mode() function for this). This approach gives you majority-vote
    predictions over the test set.

    d. Evaluate these predictions on the test set: you should obtain a slightly higher
    accuracy than your first model (about 0.5 to 1.5% higher). Congratulations,
    you have trained a Random Forest classifier!
