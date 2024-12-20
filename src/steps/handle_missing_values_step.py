import pandas as pd
from zenml import step
from src.handle_missing_values import MissingValueHandler, DropMissingValues, FillMissingValues

@step
def handle_missing_values_step(df : pd.DataFrame, strategy : str = 'drop' , axis : int = 0, fill_value = None, thresh = None):
   if strategy == 'drop':
      handler = MissingValueHandler(DropMissingValues(axis, thresh))
   elif strategy in ['mean', 'median', 'mode', 'constant']:
      handler = MissingValueHandler(FillMissingValues(strategy,fill_value))
   else :
      raise ValueError("unsupported strategy")
   
   return handler.handle_missing_value(df)
    