from zenml import Model , pipeline, step
from ..steps.data_ingestion_step import data_ingestion_step
from ..steps.handle_missing_values_step import handle_missing_values_step
from ..steps.data_splitter_step import data_splitter_step

@pipeline(
    model = Model(
        name = "prices_predictor"
    ),
)

#def ml_pipeline():
#@pipeline(enable_cache=False, model=Model(name="prices_predictor"))
def ml_pipeline():
    # Step 1: Data Ingestion
    # - Reads data from zip file
    # - Uses Factory pattern for data ingestion
    # - Returns raw dataframe
    raw_data = data_ingestion_step(file_path="data/archive.zip", ext=".zip")

    # Step 2: Data Splitting
    # - Splits data into train and test sets
    # - Returns X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = data_splitter_step(
        raw_data, target_column="sale_price"
    )


    # Step 3: Handle Missing Values
    # - Uses Strategy pattern for handling missing values
    # - Default strategy is 'mean' for numerical columns
    # - Returns cleaned dataframe
#    X_train_fill_data = handle_missing_values_step(X_train)

    # Step 4: Feature Engineering
    # - Applies log transformation to selected features
    # - Transforms 'ground_living_area' and 'sale_price'
    # - Returns engineered dataframe
#    engineer_data = feature_engineering_step(X_train_fill_data)

    # Step 5: Outlier Detection and Handling
    # - Uses Z-score method for outlier detection
    # - Handles outliers in 'sale_price' column
    # - Returns cleaned dataframe
#    X_train_clean_data = outlier_detection_step(engineer_data, column="sale_price")

    # Step 6: Model Building
    # - Uses MLflow for experiment tracking
    # - Implements pipeline with StandardScaler and LinearRegression
    # - Handles categorical encoding
    # - Returns trained model
 #   trained_model = model_building_step(X_train_clean_data, y_train)

    # Step 7: Model Evaluation
    # - Calculates mean squared error and R2 score
    # - Logs metrics to MLflow
    # - Returns evaluation metrics
 #   model_evaluator_step(trained_model, X_test, y_test)
