from zenml import step
from src.model_building import ModelBuilder, LinearRegressionStrategy, RandomForestStrategy
import pandas as pd
@step
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) :
      # Create a new ModelBuilder instance with the LinearRegressionStrategy
      builder = ModelBuilder(RandomForestStrategy())
      # Build and train the model
      trained_model = builder.build_model(X_train, y_train)
      return trained_model
