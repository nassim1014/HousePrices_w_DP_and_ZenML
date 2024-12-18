from abc import abstractmethod, ABC
import pandas as pd
import zipfile
import os

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path):
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path) -> pd.DataFrame:
        if not file_path.endswith('.zip'):
            return "This is not a .zip file"

        # Verify the file is a valid zip file
        if not zipfile.is_zipfile(file_path):
            return "This is not a valid zip file"
        
        # Open the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List all files in the zip archive
            file_list = zip_ref.namelist()

            # Filter for CSV files
            csv_files = [f for f in file_list if f.endswith('.csv')]

            # If no CSV files found
            if not csv_files:
                return "No CSV files found in the zip file."
                       # If there is more than one CSV file
            if len(csv_files) > 1:
                return "There are multiple CSV files in the zip file."

            # If there is exactly one CSV file, read it as a DataFrame
            csv_file_name = csv_files[0]
            with zip_ref.open(csv_file_name) as csv_file:
                df = pd.read_csv(csv_file)
                return df

class CSVDataIngestor(DataIngestor):
    def ingest(file_path) -> pd.DataFrame:
        # Check if the file is a CSV
        if not file_path.endswith('.csv'):
            return "This is not a .csv file"

        # Check if the file exists
        if not os.path.isfile(file_path):
            return "The file does not exist."

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            return f"An error occurred while reading the CSV file: {str(e)}"

class JSONDataIngestor:
    def ingest(json_file_path) -> pd.DataFrame:
        # Check if the file is a JSON
        if not json_file_path.endswith('.json'):
            return "This is not a .json file"

        # Check if the file exists
        if not os.path.isfile(json_file_path):
            return "The file does not exist."

        try:
            # Read the JSON file into a DataFrame
            df = pd.read_json(json_file_path)
            return df
        except ValueError as e:
            return f"An error occurred while reading the JSON file: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extention:str):
        if file_extention == "json":
            return JSONDataIngestor()
        elif file_extention == ".csv":
            return CSVDataIngestor()
        elif file_extention == ".zip" :
            return ZipDataIngestor()
        else :
            raise ValueError(f"no ingestor for file extension : {file_extention}")