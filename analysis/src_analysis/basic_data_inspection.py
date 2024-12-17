from abc import ABC, abstractmethod
import pandas as pd

# 1. Abstract Strategy Interface
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass


# 2. Concrete Strategy: Inspect Data Types and Non-Null Counts
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("### Data Types and Non-Null Counts ###")
        return df.info()  # Outputs data types, non-null counts


# 3. Concrete Strategy: Summary Statistics for Numerical Columns
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("### Summary Statistics (Numerical Columns) ###")
        return df.describe()  # Outputs summary statistics


# 4. Concrete Strategy: Summary for Categorical Columns
class CategoricalSummaryInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("### Summary Statistics (Categorical Columns) ###")
        return df.describe(include='object')  # Outputs summary for categorical columns


# 5. Context Class: DataInspector
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy  # Default strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Change the strategy dynamically.
        """
        self._strategy = strategy

    def inspect_data(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.
        """
        self._strategy.inspect(df)


# Example Usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Salary': [50000, 60000, 70000, 80000],
        'Department': ['HR', 'IT', 'Finance', 'IT']
    }
    df = pd.DataFrame(data)

    # Initialize the context with a default strategy
    inspector = DataInspector(DataTypesInspectionStrategy())

    # Strategy 1: Inspect Data Types and Non-Null Counts
    inspector.inspect_data(df)

    # Switch to Strategy 2: Summary Statistics for Numerical Columns
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.inspect_data(df)

    # Switch to Strategy 3: Summary for Categorical Columns
    inspector.set_strategy(CategoricalSummaryInspectionStrategy())
    inspector.inspect_data(df)
