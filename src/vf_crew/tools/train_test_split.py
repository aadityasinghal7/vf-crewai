"""Train-Test Split Tool for validation."""

from typing import Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import math


class TrainTestSplitInput(BaseModel):
    """Input schema for TrainTestSplitTool."""
    time_series_data: Dict = Field(..., description="Prophet-formatted time series data")
    train_ratio: float = Field(default=0.8, description="Proportion of data for training (default: 0.8)")


class TrainTestSplitTool(BaseTool):
    name: str = "Train-Test Split Tool"
    description: str = (
        "Performs temporal train-test split on time series data. "
        "Splits data chronologically to avoid data leakage. "
        "Default: 80% training, 20% testing."
    )
    args_schema: Type[BaseModel] = TrainTestSplitInput

    def _run(self, time_series_data: Dict[str, Any], train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Split time series data temporally for validation.

        Args:
            time_series_data: Dictionary containing 'data' key with Prophet DataFrame
            train_ratio: Proportion of data for training (0 to 1)

        Returns:
            Dictionary containing:
                - train_data: Training set
                - test_data: Testing set
                - split_info: Information about the split
        """
        try:
            # Extract data
            if isinstance(time_series_data, dict) and 'data' in time_series_data:
                df = time_series_data['data'].copy()
            else:
                df = time_series_data.copy()

            # Validate input
            if not isinstance(df, pd.DataFrame):
                return {
                    "success": False,
                    "error": "Input must be a pandas DataFrame",
                    "train_data": None,
                    "test_data": None
                }

            if 'ds' not in df.columns or 'y' not in df.columns:
                return {
                    "success": False,
                    "error": "DataFrame must have 'ds' and 'y' columns",
                    "train_data": None,
                    "test_data": None
                }

            # Sort by date
            df = df.sort_values('ds').reset_index(drop=True)

            # Calculate split point
            n_total = len(df)
            n_train = math.floor(n_total * train_ratio)

            # Ensure at least some test data
            if n_train >= n_total - 1:
                n_train = n_total - 2  # Leave at least 2 points for testing

            # Ensure minimum training data
            if n_train < 10:
                return {
                    "success": False,
                    "error": f"Insufficient training data after split: {n_train} points (minimum: 10)",
                    "train_data": None,
                    "test_data": None
                }

            # Perform split
            train_data = df.iloc[:n_train].copy()
            test_data = df.iloc[n_train:].copy()

            # Split information
            split_info = {
                'total_samples': n_total,
                'train_samples': n_train,
                'test_samples': len(test_data),
                'train_ratio_actual': n_train / n_total,
                'test_ratio_actual': len(test_data) / n_total,
                'train_start': train_data['ds'].min().strftime('%Y-%m-%d'),
                'train_end': train_data['ds'].max().strftime('%Y-%m-%d'),
                'test_start': test_data['ds'].min().strftime('%Y-%m-%d'),
                'test_end': test_data['ds'].max().strftime('%Y-%m-%d'),
                'split_date': train_data['ds'].max().strftime('%Y-%m-%d')
            }

            return {
                "success": True,
                "train_data": train_data,
                "test_data": test_data,
                "split_info": split_info,
                "message": f"Split complete: {n_train} train samples, {len(test_data)} test samples"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error during train-test split: {str(e)}",
                "train_data": None,
                "test_data": None,
                "split_info": {}
            }
