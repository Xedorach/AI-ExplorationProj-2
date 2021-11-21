import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

'''
 Project 2 code for AI Explorations, EEEE 547 course Fall 2021
 Written by Saifeldin Hassan and Tony Hanna.
 
'''

def main():

    df = load_data('mall_customers.csv')
    x_train, y_train, x_test, y_test = train_model(df)

    # Data visualization

    plt.figure("Age vs Spending Score")
    sns.scatterplot(data=df, x='Age', y='Spending_Score_100')

    plt.figure("Gender vs Spending Score")
    sns.barplot(data=df, x='Gender', y='Spending_Score_100')

    plt.figure("Annual income vs Spending Score")
    sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score_100')

    # Linear regression
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(x_train, y_train)
    linear_regression_prediction = linear_regression_model.predict(x_test)
    evaluate(y_test, linear_regression_prediction, "Linear Regression")
    plot_scatter('Linear Regression model', 'Linear Regression', y_test, linear_regression_prediction)

    # Bayesian Regression
    bayesian_ridge_model = BayesianRidge()
    bayesian_ridge_model.fit(x_train, y_train)
    bayesian_regression_prediction = bayesian_ridge_model.predict(x_test)

    plot_scatter('Bayesian Regression', 'Bayesian Ridge model', y_test, bayesian_regression_prediction)
    evaluate(y_test, bayesian_regression_prediction, "Bayesian Ridge")

    # Least angle regression
    lars_lasso_model = LassoLars(alpha=.1, normalize=False)
    lars_lasso_model.fit(x_train, y_train)
    lars_lasso_prediction = lars_lasso_model.predict(x_test)

    plot_scatter("LARS LASO", "Least angle regression", y_test, lars_lasso_prediction)
    evaluate(y_test, lars_lasso_prediction, "LARS LASO")

    # Ridge regression model

    ridge_regression_model = Ridge(alpha=.5)
    ridge_regression_model.fit(x_train, y_train)
    ridge_regression_prediction = ridge_regression_model.predict(x_test)

    plot_scatter("Ridge regression", "Ridge regression", y_test, ridge_regression_prediction)
    evaluate(y_test, ridge_regression_prediction, "Ridge Regression")

    plt.show()


def load_data(filename):
    """
    Function takes a string as input, data_set.csv loads it and displays data in shell.
    also renames male and female to 0 and 1 for easier handling
   """

    df = pd.read_csv(filename)
    print("\n", df, "\n", df.info(), "\n", df.columns)
    # Courtesy of Fardin, cleaner than a big for loop to rename the entries for gender
    df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'Male' else 1)

    return df


def train_model(df):
    """
            Function preps model training variables
    """
    # Create the test train split
    train, test = train_test_split(df, test_size=0.4)
    x_train = train[['Gender', 'Age', 'Annual_Income']]
    y_train = train['Spending_Score_100']
    x_test = test[['Gender', 'Age', 'Annual_Income']]
    y_test = test['Spending_Score_100']

    return x_train, y_train, x_test, y_test


def evaluate(actual, prediction, name):
    """
            prints L1 and L2 loss

    """
    l1_loss = mean_absolute_error(prediction, actual)
    l2_loss = mean_squared_error(prediction, actual, squared=1)

    print('\n' + f'L1 Loss ( {name} )= {l1_loss:.2f}' + '\n'f'L2 Loss ( {name} ) = {l2_loss:.2f}')


def plot_scatter(figure, title, x_plot, y_plot):
    """

            To make main function cleaner, shortened scatter plot function

    """
    plt.figure(figure)
    sns.scatterplot(x=x_plot, y=y_plot)
    plt.title(title)
    plt.xlabel('Real')
    plt.ylabel('Prediction')


if __name__ == "__main__":
    main()
