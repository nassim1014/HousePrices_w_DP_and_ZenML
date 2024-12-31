import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports a random row of data from the housing dataset."""
    # Load the housing.csv dataset
    df = pd.read_csv(r"src\Data\Housing.csv")

    # Select a random row from the DataFrame
    random_row = df.sample(n=1)

    # Convert the random row to a JSON string
    json_data = random_row.to_json(orient="split")

    return json_data
