import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        pass


# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:

        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:

        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass
