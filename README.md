# Naive Bayes Classifier for Divar Dataset

This Jupyter notebook file contains an implementation of the Naive Bayes Classifier to predict the category of items in
the Divar dataset. The implementation is done using Python and the following libraries: pandas, hazm, parsivar, and
matplotlib.

## Problem Statement

The Divar dataset contains information about items being sold on an online platform. The goal is to predict the category
of each item based on its description and title.

## Approach

The approach used in this notebook consists of two main phases: preprocessing and solving.

### Phase 1: Preprocessing

The first step involves preprocessing the input data, which includes the following:

1. Stemming: to normalize the data, we used stemming to find the root form of each word by cutting off its affixes,
   regardless of context. We chose stemming over lemmatization because it is faster and produces acceptable results for
   our purposes.

2. Tokenization: we used tokenization to split the text into individual words.

### Phase 2: Solving

The next step involves creating a bag of words model to predict the category of each item based on its description and
title. We used the Naive Bayes Classifier to calculate the posterior probability of each category given the words in the
item's description and title.

We also used bigrams to capture the context of words and additive smoothing to prevent the probability of a category
from being zero.

Finally, we evaluated the model's performance usingaccuracy, precision, recall, F1 score, and macro/micro/weighted
averaging based on the number of classes and their distribution.

## Code

The code is divided into several functions:

1. `normalize(train_data)`: preprocesses the train data by tokenizing and stemming the description and title.

2. `naive_bayes_model(data)`: creates a bag of words model using the train data.

3. `naive_bayes(test, model)`: predicts the category of each item in the test data using the bag of words model.

4. `smooth_naive_bayes(test, model, alpha)`: predicts the category of each item in the test data using additive
   smoothing.

5. `analyse(out, expected, cat, table)`: calculates the precision, recall, and F1 score for a specific category.

6. `complete_analysis(output, test_data)`: calculates the accuracy, macro/micro/weighted averaging, and precision,
   recall, and F1 score for each category.

## Results

The results show that adding additive smoothing improved the accuracy of the model. However, there were still some wrong
answers, which could be due to the presence of new words in the test data or the ambiguity of the item's category.

## Conclusion

The Naive Bayes Classifier is an effective method for predicting the category of items in the Divar dataset. However,
further improvements can be made by using more advanced techniques such as deep learning or ensemblelearning.
Additionally, exploring other pre-processing techniques such as lemmatization or using different stemmers could also
improve the model's performance.

Overall, the Naive Bayes Classifier provides a simple and efficient way to solve the problem of predicting item
categories in the Divar dataset.