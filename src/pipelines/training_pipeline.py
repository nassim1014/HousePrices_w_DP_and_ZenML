from zenml import Model , pipeline, step
from src.steps.data_ingestion_step import data_ingestion_step
from src.steps.handle_missing_values_step import handle_missing_values_step
from src.steps.data_splitter_step import data_splitter_step
from src.steps.feature_engineering_step import feature_engineering_step
from src.steps.outlier_detection_step import outlier_detection_step
from src.steps.model_building_step import model_building_step
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
    raw_data = data_ingestion_step(file_path="src/Data/Housing.csv", ext=".csv")
    # Step 2: Data Splitting
    # - Splits data into train and test sets
    # - Returns X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = data_splitter_step(
        raw_data, target_column="price"
    )

    # Step 3: Handle Missing Values
    # - Uses Strategy pattern for handling missing values
    # - Default strategy is 'mean' for numerical columns
    # - Returns cleaned dataframe
 #   X_train_fill_data = handle_missing_values_step(X_train)

    # Step 4: Feature Engineering
    # - Applies log transformation to selected features
    # - Transforms 'ground_living_area' and 'sale_price'
    # - Returns engineered dataframe
 #   engineer_data = feature_engineering_step(df = X_train_fill_data, features = ['area'], strategy = 'log')

    # Step 5: Outlier Detection and Handling
    # - Uses Z-score method for outlier detection
    # - Handles outliers in 'sale_price' column
    # - Returns cleaned dataframe
 #   X_train_clean_data = outlier_detection_step(engineer_data, column_name="area")
    # Step 6: Model Building
    # - Uses MLflow for experiment tracking
    # - Implements pipeline with StandardScaler and LinearRegression
    # - Handles categorical encoding
    # - Returns trained model
    trained_model = model_building_step(X_train, y_train)

    # Step 7: Model Evaluation
    # - Calculates mean squared error and R2 score
    # - Logs metrics to MLflow
    # - Returns evaluation metrics
    model_evaluator_step(trained_model, X_test, y_test)
