#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
from dotenv import load_dotenv

from vf_crew.crew import VfCrew

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.


def get_top_n_skus():
    """
    Get top_n_skus value from environment variable or use default.

    Returns:
        int: Number of top SKUs to process (default: 10)

    Raises:
        ValueError: If TOP_N_SKUS environment variable is not a valid positive integer
    """
    default_value = 10
    env_value = os.environ.get('TOP_N_SKUS')

    if env_value is None:
        return default_value

    try:
        value = int(env_value)
        if value <= 0:
            raise ValueError(f"TOP_N_SKUS must be a positive integer, got: {value}")
        return value
    except ValueError as e:
        raise ValueError(f"Invalid TOP_N_SKUS environment variable: {env_value}. Must be a positive integer.") from e


def run():
    """
    Run the volume forecasting crew.
    """
    # Prompt user for input file path
    print("\nPlease enter the path to your input Excel file:")
    print("(Press Enter to use default: data/input/TBS Weekly Volume.xlsx)")
    user_input = input("> ").strip()

    if user_input:
        input_file_path = os.path.abspath(user_input)
    else:
        input_file_path = os.path.abspath("data/input/TBS Weekly Volume.xlsx")

    # Fixed output path
    output_file_path = os.path.abspath("data/output/volume_forecasts.xlsx")

    inputs = {
        'file_path': input_file_path,
        'output_path': output_file_path,
        'top_n_skus': 10,  # Number of top SKUs to process by volume
        'min_weeks': 10,  # Minimum weeks of data required for valid time series
        'forecast_periods': 26,  # 6 months = 26 weeks
        'n_weeks': 151,  # Total weeks in dataset
    }

    print(f"\n{'='*80}")
    print(f"Volume Forecasting Crew - Starting Execution")
    print(f"{'='*80}")
    print(f"Input File: {input_file_path}")
    print(f"Output File: {output_file_path}")
    print(f"Top N SKUs: {inputs['top_n_skus']} (by volume)")
    print(f"Minimum Weeks Required: {inputs['min_weeks']}")
    print(f"Forecast Horizon: {inputs['forecast_periods']} weeks (6 months)")
    print(f"{'='*80}\n")

    try:
        result = VfCrew().crew().kickoff(inputs=inputs)
        print(f"\n{'='*80}")
        print(f"Forecasting Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file_path}")
        print(f"{'='*80}\n")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    # Prompt user for input file path
    print("\nPlease enter the path to your input Excel file:")
    print("(Press Enter to use default: data/input/TBS Weekly Volume.xlsx)")
    user_input = input("> ").strip()

    if user_input:
        input_file_path = os.path.abspath(user_input)
    else:
        input_file_path = os.path.abspath("data/input/TBS Weekly Volume.xlsx")

    # Fixed output path
    output_file_path = os.path.abspath("data/output/volume_forecasts.xlsx")

    inputs = {
        'file_path': input_file_path,
        'output_path': output_file_path,
        'top_n_skus': 10,
        'min_weeks': 10,
        'forecast_periods': 26,
        'n_weeks': 151,
    }

    try:
        VfCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        VfCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    # Prompt user for input file path
    print("\nPlease enter the path to your input Excel file:")
    print("(Press Enter to use default: data/input/TBS Weekly Volume.xlsx)")
    user_input = input("> ").strip()

    if user_input:
        input_file_path = os.path.abspath(user_input)
    else:
        input_file_path = os.path.abspath("data/input/TBS Weekly Volume.xlsx")

    # Fixed output path
    output_file_path = os.path.abspath("data/output/volume_forecasts.xlsx")

    inputs = {
        'file_path': input_file_path,
        'output_path': output_file_path,
        'top_n_skus': 10,  # Number of top SKUs to process by volume
        'min_weeks': 10,
        'forecast_periods': 26,
        'n_weeks': 151,
    }

    try:
        VfCrew().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Extract parameters from trigger payload or use defaults
    file_path = trigger_payload.get('file_path', 'data/input/TBS Weekly Volume.xlsx')
    output_path = trigger_payload.get('output_path',
                                      f'data/output/volume_forecasts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "file_path": file_path,
        "output_path": output_path,
        "top_n_skus": trigger_payload.get('top_n_skus', 10),  # Number of top SKUs to process by volume
        "min_weeks": trigger_payload.get('min_weeks', 10),
        "forecast_periods": trigger_payload.get('forecast_periods', 26),
        "n_weeks": trigger_payload.get('n_weeks', 151),
    }

    try:
        result = VfCrew().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
