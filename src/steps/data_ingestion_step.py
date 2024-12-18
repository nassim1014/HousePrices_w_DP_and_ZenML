   # Handles data loading from zip files using Factory pattern
   # Returns raw dataframe 
import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step
@step
def data_ingestion_step(file_path : str, ext : str) -> pd.DataFrame :
    data_ingestor = DataIngestorFactory.get_data_ingestor(ext)
    return data_ingestor.ingest(file_path)