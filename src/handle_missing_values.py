from abc import ABC , abstractmethod
import logging
import pandas as pd

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class MissingValuesHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame :
        pass

class DropMissingValues(MissingValuesHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df : pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis} and threshold={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values are dropped")
        return df_cleaned

class FillMissingValues(MissingValuesHandlingStrategy):
    def __init__(self, method='mean', fill_value=None):
        """
        Initialize with filling method.
        
        Args:
            method (str): Method to use ('mean', 'median', 'mode', 'constant')
            fill_value: Value to use when method is 'constant'
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")
        
        # Create a copy of the dataframe
        df_filled = df.copy()
        
        if self.method == 'mean':
            # Select only numeric columns
            numeric_columns = df_filled.select_dtypes(include=['float64', 'int64']).columns
            df_filled[numeric_columns] = df_filled[numeric_columns].fillna(df_filled[numeric_columns].mean())
            
        elif self.method == 'median':
            numeric_columns = df_filled.select_dtypes(include=['float64', 'int64']).columns
            df_filled[numeric_columns] = df_filled[numeric_columns].fillna(df_filled[numeric_columns].median())
            
        elif self.method == 'mode':
            for column in df_filled.columns:
                df_filled[column] = df_filled[column].fillna(df_filled[column].mode()[0])
                
        elif self.method == 'constant' and self.fill_value is not None:
            df_filled = df_filled.fillna(self.fill_value)
            
        else:
            logging.error(f"Unknown method: {self.method}. No missing values handled.")
            return df
        
        logging.info("Missing values filled")
        return df_filled

class MissingValueHandler:
    def __init__(self, missing_value_strategy : MissingValuesHandlingStrategy):
        self._missing_value_strategy = missing_value_strategy

    def set_missing_value_strategy(self, missing_value_strategy : MissingValuesHandlingStrategy):
        self._missing_value_strategy = missing_value_strategy
    
    def handle_missing_value(self , df: pd.DataFrame) -> pd.DataFrame:
        return self._missing_value_strategy.handle(df)