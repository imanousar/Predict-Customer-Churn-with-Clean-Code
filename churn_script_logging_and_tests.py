"""
Project: Churn Prediction
Date:   8/4/2023
Author: Giannis Manousaridis
"""

import os
import glob
import logging
import warnings
import pandas as pd
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    Test data import
    """
    try:
        df = import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    Test perform eda function
    """

    # Preprocess
    try:
        df = cls.import_data("data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
                     lambda val: 0 if val == "Existing Customer" else 1)
    except Exception as err:
        logging.error("Testing perform_eda: Preprocess data wasn't possible")
        raise err

    # Function Execution
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: The function wasn't executed")
        raise err

    # Folder Exists
    try:
        assert os.path.exists('images/eda')
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: The folder doesn't exist.")
        raise err

    # Folder contains   images
    try:
        assert glob.glob(os.path.join('images/eda', '*.png'))
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: The folder doesn't have png images.")
        raise err


def test_encoder_helper(encoder_helper):
    """
    Test encoder helper
    """
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    response = 'Churn'
    headers = [header+'_Churn' for header in cat_columns]

    # Preprocess
    try:
        df = cls.import_data("data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
                     lambda val: 0 if val == "Existing Customer" else 1)
    except Exception as err:
        logging.error("Testing perform_eda: Preprocess data wasn't possible")
        raise err

    # Function Execution
    try:
        data = encoder_helper(df, cat_columns, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: The function wasn't executed")
        raise err

    # Created columns exist and are numerical
    try:
        assert set(headers).issubset(set(data.columns))
        assert any(data[headers].select_dtypes(include=['number']
                                               ).columns.tolist())
    except Exception as err:
        logging.error("Testing encoder_helper: The columns are not right")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering
    """
    # Preprocess
    try:
        df = cls.import_data("data/bank_data.csv")
        response = 'Churn'
        df[response] = df['Attrition_Flag'].apply(
                     lambda val: 0 if val == "Existing Customer" else 1)
    except Exception as err:
        logging.error("Testing perform_feature_engineering: Preprocess data wasn't possible")
        raise err

    # Function Execution
    try:
        x_data, x_train, x_test, y_train, y_test = perform_feature_engineering(
                                                                df, response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: The function wasn't executed")
        raise err

    # Return values are not empty and are np.ndarrays
    try:
        assert type(x_data) == pd.DataFrame
        assert type(x_train) == pd.DataFrame
        assert type(x_test) == pd.DataFrame
        assert type(y_train) == pd.Series
        assert type(y_test) == pd.Series

    except AssertionError as err:
        logging.error(type(y_train))
        logging.error(
                    "Testing perform_feature_engineering: Wrong return values")
        raise err


def test_train_models(train_models):
    """
    Test train_models
    """

    # Preprocess
    try:
        df = cls.import_data("data/bank_data.csv")
        response = 'Churn'
        df[response] = df['Attrition_Flag'].apply(
                     lambda val: 0 if val == "Existing Customer" else 1)

        x_data, x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, response)
    except Exception as err:
        logging.error("Testing test_train_models: Preprocess data wasn't possible")
        raise err

    # Function Execution
    try:
        train_models(x_data, x_train, x_test, y_train, y_test)
        logging.info("Testing test_train_models: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing test_train_models: The function wasn't executed")
        raise err

    # Folder reports exists
    try:
        assert os.path.exists('images/reports')
    except FileNotFoundError as err:
        logging.error(
            "Testing test_train_models: The folder reports doesn't exist.")
        raise err

    # Folder reports contains images
    try:
        assert glob.glob(os.path.join('images/reports', '*.png'))
    except FileNotFoundError as err:
        logging.error(
            "Testing test_train_models: The folder reports doesn't have png images.")
        raise err

    # Folder results exists
    try:
        assert os.path.exists('images/results')
    except FileNotFoundError as err:
        logging.error(
            "Testing test_train_models: The folder results doesn't exist.")
        raise err

    # Folder results contains images
    try:
        assert glob.glob(os.path.join('images/results', '*.png'))
    except FileNotFoundError as err:
        logging.error(
            "Testing test_train_models: The folder results doesn't have png images.")
        raise err

    # Folder models exists
    try:
        assert os.path.exists('models')
    except FileNotFoundError as err:
        logging.error(
            "Testing test_train_models: The folder models doesn't exist.")
        raise err

    # Folder models contains models
    try:
        assert glob.glob(os.path.join('models', '*.pkl'))
    except FileNotFoundError as err:
        logging.error(
            "Testing test_train_models: The folder models doesn't have models.")
        raise err


if __name__ == "__main__":

    # ignore all caught warnings
    warnings.filterwarnings("ignore")

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)