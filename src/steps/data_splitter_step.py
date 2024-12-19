   # Handles train-test splitting
   # Returns X_train, X_test, y_train, y_test 
from src.data_splitter import DataSplitterContext , SimpleTrainTestSplit
import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.base import TransformerMixin

@step
def data_splitter_step(
    df: pd.DataFrame,
    target_column: str,
    strategy: str = "simple_train_test"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    if strategy == "simple_train_test":
        splitter = SimpleTrainTestSplit()
    else:
        raise ValueError(f"Unknown splitting strategy: {strategy}")
    
    context = DataSplitterContext(splitter)
    X_train, X_test, y_train, y_test = context.split_data(df, target_column)
    
    return X_train, X_test, y_train, y_test
