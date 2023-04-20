"""
Project: Churn Prediction
Date:   8/4/2023
Author: Giannis Manousaridis
"""

import os
import warnings
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# import libraries
# plot_roc_curve has been removed in version 1.2.
# From 1.2, use RocCurveDisplay instead
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


def main() -> None:
    """
    Main function to execute
    """

    # ignore all caught warnings
    warnings.filterwarnings("ignore")

    sns.set()

    path = 'data/bank_data.csv'
    data = import_data(path)

    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(data)

    x_data, x_train, x_test, y_train, y_test = perform_feature_engineering(
                                                                       data,
                                                                       'Churn')

    train_models(x_data, x_train, x_test, y_train, y_test)


def import_data(path: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at pth

    Parameters:
        path (str): a path to the csv
    Returns:
        data (pd.DataFrame):  data in dataframe format
    """
    data = pd.read_csv(path)

    return data


def perform_eda(data: pd.DataFrame) -> None:
    """
    Perform eda on input data and save figures to images folder

    Parameters:
        data (pd.DataFrame): data to perform EDA

    Returns:
        None (None): None
    """

    # create the directory if it doesn't exist
    if not os.path.exists('images/eda'):
        os.makedirs('images/eda')

    # Create size of image
    plt.figure(figsize=(20, 10))

    # Create target variable's histogram
    data['Churn'].hist()
    plt.savefig('images/eda/churn_histogram.png')
    plt.clf()

    # Create customer's age histogram
    data['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_histogram.png')
    plt.clf()

    # Create marital's status barplot histogram
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_barplot.png')
    plt.clf()

    # Create total transactions histogram
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/total_transitions_count_histogram.png')
    plt.clf()

    # Create correlation matrix for numeric featuress
    categorical_colums_list = get_categorical_values(data)
    data_with_numeric_features = data.loc[:, ~data.columns.isin(
                                      categorical_colums_list)]

    sns.heatmap(data_with_numeric_features.corr(), annot=False, cmap='Dark2_r',
                linewidths=2)
    plt.savefig('images/eda/correlation_matrix.png')
    plt.clf()


def perform_feature_engineering(data: pd.DataFrame, response: str
                                ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                           pd.Series, pd.Series]:
    """
    Parameters:
        data (pd.DataFrame): data
        response (str): string of response name [optional argument that could
                        be used for naming variables or index y column]

    Returns:
        x_train (pd.DataFrame): X training data
        x_test (pd.DataFrame): X testing data
        y_train (pd.Series): y training data
        y_test (pd.Series): y testing data
    """

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    data = encoder_helper(data, cat_columns, response)

    y_target = data[response]
    x_data = pd.DataFrame()

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit',
                 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                 'Avg_Utilization_Ratio', 'Gender_Churn',
                 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    x_data[keep_cols] = data[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_target,
                                                        test_size=0.3,
                                                        random_state=42)

    return x_data, x_train, x_test, y_train, y_test


def encoder_helper(temp_df: pd.DataFrame, category_list: List,
                   response: str) -> pd.DataFrame:
    """
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category.

    Parameters:
        temp_df (pd.DataFrame): train data
        category_lst (List): list of columns that contain categorical features
        response (str): string of response name [optional argument that could 
                        be used for naming variables or index y column]

    Returns:
        temp_df (pd.DataFrame): data after encoding
    """
    for column_name in category_list:

        # Map categorical values with numerical values using mean
        group_values = temp_df.groupby(column_name)[response].mean()

        # Define the name of the new numerical column
        temp_name = column_name+'_'+response

        # Enrich the new column with data
        temp_df[temp_name] = temp_df[column_name].map(group_values)

    return temp_df


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr: np.ndarray,
                                y_train_preds_rf: np.ndarray,
                                y_test_preds_lr: np.ndarray,
                                y_test_preds_rf: np.ndarray) -> None:
    """
    Produces classification report for training and testing results and stores
    report as image in images folder

    Parameters:
        y_train (pd.Series): training response values
        y_test (pd.Series):  test response values
        y_train_preds_lr (np.ndarray): training predictions from logistic 
                                       regression
        y_train_preds_rf (np.ndarray): training predictions from random forest
        y_test_preds_lr (np.ndarray): test predictions from logistic regression
        y_test_preds_rf (np.ndarray): test predictions from random forest

    Returns:
        None
    """

    # create the directory if it doesn't exist
    if not os.path.exists('images/reports'):
        os.makedirs('images/reports')

    plt.figure(figsize=(8, 6))
    plt.text(0.01, 1.1, str('Random Forest Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.15, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/reports/RF_report.png')
    plt.clf()  # clear the figure

    plt.figure(figsize=(8, 6))
    plt.text(0.01, 1.1, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.15, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')

    plt.savefig('images/reports/LR_report.png')
    plt.clf()


def feature_importance_plot(model, x_test: np.ndarray, x_data: pd.DataFrame,
                            return_pth: str) -> None:
    """
    creates and stores the feature importances in pth
    Parameters:
        model (sklearn-model): model object containing feature_importances_
        x_test (pd.DataFrame): X test data
        x_data (np.ndarray): X data
        return_pth (str): path to store the figure

    Return:
             None
    """

    plt.clf()
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_test)

    # By default `summary_plot` calls `plt.show()` to ensure the plot displays.
    # If you pass `show=False` to `summary_plot` then it won't, and
    # that might fix the blank figure issue.
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)

    if not os.path.exists(return_pth):
        os.makedirs(return_pth)

    plt.savefig(return_pth + 'best_estimator.png')

    plt.clf()

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(return_pth + 'feature_importance_barplots.png')
    plt.clf()


def train_models(x_data: pd.DataFrame, x_train: pd.DataFrame,
                 x_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
    """
    Train, store model results: images + scores, and store models
    Parameters:
        x_data (pd.DataFrame): X data
        x_train (pd.DataFrame): X training data
        x_test (pd.DataFrame): X testing data
        y_train (pd.Series): y training data
        y_test (pd.Series): y testing data
    Return:
        None (None): None
    """

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=2)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # Plots

    # create the directory if it doesn't exist
    if not os.path.exists('images/results'):
        os.makedirs('images/results')

    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    plt.savefig('images/results/roc_curve_logRegression.png')

    plt.figure(figsize=(15, 8))
    ax_plot = plt.gca()
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_,
                                   x_test, y_test, ax=ax_plot, alpha=0.8)
    lrc_plot.plot(ax=ax_plot, alpha=0.8)

    plt.savefig('images/results/roc_curve_LR_vs_RF.png')

    feature_importance_plot(cv_rfc, x_test, x_data, 'images/results/')

    # save best model

    # create the directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')


def get_categorical_values(data: pd.DataFrame) -> List:
    """
    Get  a list with the names of categorical values of a dataframe

    Parameters:
        data (pd.DataFrame): input dataframe

    Return:
        categorical_columns (List): list with the names of categorical values
    """
    categorical_columns = data.select_dtypes(include=['object'
                                                      ]).columns.tolist()

    return categorical_columns


if __name__ == "__main__":

    main()
