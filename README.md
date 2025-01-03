# House Price Prediction with MLOps

## Project Overview
An end-to-end machine learning project implementing house price prediction with robust MLOps practices. The project demonstrates how implementation quality can differentiate a simple project, focusing on core ML principles and MLOps integration .

## Key Features
- End-to-end ML pipeline implementation
- MLOps integration using ZenML and MLflow
- Comprehensive data analysis and feature engineering
- Model training, evaluation, and deployment
- CI/CD pipeline integration
- Experiment tracking and monitoring 

## Tech Stack
- **Orchestration**: ZenML
- **Experiment Tracking**: MLflow
- **Model Deployment**: MLflow
- **Data Analysis**: Pandas, NumPy, Seaborn
- **ML Framework**: Scikit-learn
- **Design Patterns**: Strategy, Template, Factory 

## Project Structure
```
├── source/
│   ├── ingest_data.py         # Data ingestion implementation
│   ├── handle_missing_values.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
├── steps/
│   ├── data_ingestion.py
│   ├── handle_missing_values.py
│   └── model_building.py
├── pipelines/
│   ├── training_pipeline.py
│   └── deployment_pipeline.py
├── data/
│   └── archive.zip
```

## Installation & Setup
1. Install required dependencies
2. Set up ZenML with MLflow stack
3. Configure MLflow tracking server
4. Ensure proper stack configuration:
   - Orchestrator: Default
   - Experiment Tracker: MLflow
   - Model Deployer: MLflow
   - Artifact Store: Default 

## Pipeline Steps
1. Data Ingestion
2. Missing Value Analysis & Handling
3. Feature Engineering
4. Outlier Detection
5. Data Splitting
6. Model Building
7. Model Evaluation
8. Model Deployment 

## Design Patterns Used
- **Factory Pattern**: For data ingestion
- **Strategy Pattern**: For data inspection and analysis
- **Template Pattern**: For missing value analysis 

## Model Deployment
- Continuous deployment pipeline for model training and deployment
- Inference pipeline for batch predictions
- MLflow model serving
- REST API endpoint for predictions 

## Future Scope
- Experiment with different scaling techniques
- Test additional model assumptions
- Implement more feature engineering techniques
- Add more comprehensive testing
- Enhance data processing techniques 

## Usage
```python
# Run training pipeline
python run_pipeline.py

# Run deployment
python deployment.py

# Make predictions using the deployed model
python sample_predict.py
```