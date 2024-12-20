'''
import click
from src.pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline()
    run.run()
    # You can uncomment and customize the following lines if you want to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]  # Replace with actual step name if different
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()

'''

from zenml import step, pipeline

# Step 1: Generate a random number
@step
def generate_number() -> int:
    import random
    number = random.randint(1, 100)
    print(f"Generated number: {number}")
    return number

# Step 2: Double the number
@step
def double_number(input_number: int) -> int:
    doubled = input_number * 2
    print(f"Doubled number: {doubled}")
    return doubled

# Step 3: Print the result
@step
def print_result(doubled_number: int) -> None:
    print(f"Final result: {doubled_number}")

# Define the pipeline
@pipeline
def simple_pipeline():
    number = generate_number()
    doubled = double_number(input_number=number)
    print_result(doubled_number=doubled)

# Run the pipeline
if __name__ == "__main__":
    simple_pipeline()