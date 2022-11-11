import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pickle
import tarfile
from typing import Any, Dict

import fsspec
from sagemaker.sklearn.estimator import SKLearn
from sklearn.linear_model import LinearRegression


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters['data_science']['active_modelling_pipeline']['model_options']['features']]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters['data_science']['active_modelling_pipeline']['model_options']['test_size']
        , random_state=parameters['data_science']['active_modelling_pipeline']['model_options']['random_state']
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)

def train_model_sagemaker(
    X_train_path: str, sklearn_estimator_kwargs: Dict[str, Any]
) -> str:
    """Train the linear regression model on SageMaker.

    Args:
        X_train_path: Full S3 path to `X_train` dataset.
        sklearn_estimator_kwargs: Keyword arguments that will be used
            to instantiate SKLearn estimator.

    Returns:
        Full S3 path to `model.tar.gz` file containing the model artifact.

    """
    sklearn_estimator = SKLearn(**sklearn_estimator_kwargs)

    # we need a path to the directory containing both
    # X_train (feature table) and y_train (target variable)
    inputs_dir = X_train_path.rsplit("/", 1)[0]
    inputs = {"train": inputs_dir}

    # wait=True ensures that the execution is blocked
    # until the job finishes on SageMaker
    sklearn_estimator.fit(inputs=inputs, wait=True)

    training_job = sklearn_estimator.latest_training_job
    job_description = training_job.describe()
    model_path = job_description["ModelArtifacts"]["S3ModelArtifacts"]
    return model_path


def untar_model(model_path: str) -> LinearRegression:
    """Unarchive the linear regression model artifact produced
    by the training job on SageMaker.

        Args:
            model_path: Full S3 path to `model.tar.gz` file containing
                the model artifact.

        Returns:
            Trained model.

    """
    with fsspec.open(model_path) as s3_file, tarfile.open(
        fileobj=s3_file, mode="r:gz"
    ) as tar:
        # we expect to have only one file inside the `model.tar.gz` archive
        filename = tar.getnames()[0]
        model_obj = tar.extractfile(filename)
        return pickle.load(model_obj)