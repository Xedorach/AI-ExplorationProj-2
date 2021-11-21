import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    x_train, x_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(x_train, y_train)
    predictions = model.predict(x_test)
    L1, L2 = evaluate(y_test, predictions)

    # Print results
    print(f"L1 Loss: {L1:.2f}")
    print(f"L2 Loss: {L2:.2f}")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Gender, an integer 0 (if male) and 1 (if female)
        - Age, an integer
        - Annual_Income, an integer

    labels should be the corresponding list of labels, where each label
    is the Spending Score, an integer.
    """
    df = pd.read('shopping.csv')




    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted linear regression model trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (L1, L2).

    `L1` should be a floating-point value representing the average 
    absolute difference between observed and predicted outcomes.

    `L2` should be a floating-point value representing the squared 
    difference between the observed actual outcome values and the values 
    predicted.
    
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
