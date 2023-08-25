import matplotlib.pyplot as plt
import scipy
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
import seaborn as sns
def train_logistic_regression(X, y, model, test_size=0.2):
    """
    Trains a logistic regression model on the input data and prints the confusion matrix.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target variable.
        model (LogisticRegression): An instance of LogisticRegression from scikit-learn.
        test_size (float): Proportion of the data to reserve for testing. Default is 0.2.

    Returns:
        None.

    Raises:
        TypeError: If model is not an instance of LogisticRegression.
        ValueError: If test_size is not between 0 and 1.

    """
    # Check if model is an instance of LogisticRegression
    if not isinstance(model, LogisticRegression):
        raise TypeError("model must be an instance of LogisticRegression.")

    # Check if test_size is between 0 and 1
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Fit the model to the training data and make predictions on the test data
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Print the confusion matrix
    confusion_matrix(y_test, y_pred)


def confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots a confusion matrix for the predicted vs. actual labels and prints the accuracy score.

    Args:
        y_test (np.ndarray): Actual labels of the test set.
        y_pred (np.ndarray): Predicted labels of the test set.

    Returns:
        None.

    Raises:
        TypeError: If either y_test or y_pred are not numpy arrays.

    """
    # Check if y_test and y_pred are numpy arrays
    if not isinstance(y_test, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_test and y_pred must be numpy arrays.")

    # Calculate the confusion matrix and f1 score
    cm = metrics.confusion_matrix(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='macro')

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()