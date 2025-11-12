"""Metrics Calculator Tool for validation assessment."""

from typing import Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculatorInput(BaseModel):
    """Input schema for MetricsCalculatorTool."""
    actual_values: Any = Field(..., description="Actual values (y_true)")
    predicted_values: Any = Field(..., description="Predicted values (y_pred)")


class MetricsCalculatorTool(BaseTool):
    name: str = "Metrics Calculator Tool"
    description: str = (
        "Calculates comprehensive validation metrics for forecast accuracy. "
        "Computes MAE, RMSE, MAPE, R\u00b2, and forecast bias. "
        "Handles edge cases and provides detailed metric explanations."
    )
    args_schema: Type[BaseModel] = MetricsCalculatorInput

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        Handles zero values by masking them out.
        """
        # Mask zero values to avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.nan  # Cannot calculate MAPE if all actuals are zero

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _calculate_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate forecast bias (mean error).
        Positive bias = over-forecasting, Negative bias = under-forecasting.
        """
        return np.mean(y_pred - y_true)

    def _run(self, actual_values: Any, predicted_values: Any) -> Dict[str, Any]:
        """
        Calculate comprehensive validation metrics.

        Args:
            actual_values: Actual/true values
            predicted_values: Predicted/forecast values

        Returns:
            Dictionary containing all metrics
        """
        try:
            # Convert to numpy arrays
            if isinstance(actual_values, pd.Series) or isinstance(actual_values, pd.DataFrame):
                y_true = actual_values.values.flatten()
            else:
                y_true = np.array(actual_values).flatten()

            if isinstance(predicted_values, pd.Series) or isinstance(predicted_values, pd.DataFrame):
                y_pred = predicted_values.values.flatten()
            else:
                y_pred = np.array(predicted_values).flatten()

            # Validate lengths match
            if len(y_true) != len(y_pred):
                return {
                    "success": False,
                    "error": f"Length mismatch: actual={len(y_true)}, predicted={len(y_pred)}",
                    "metrics": {}
                }

            # Check for sufficient data
            if len(y_true) < 2:
                return {
                    "success": False,
                    "error": "Insufficient data points for metric calculation (minimum: 2)",
                    "metrics": {}
                }

            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = self._calculate_mape(y_true, y_pred)
            bias = self._calculate_bias(y_true, y_pred)

            # R² score (handle edge case where variance is 0)
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = np.nan

            # Additional useful metrics
            mean_actual = np.mean(y_true)
            mean_predicted = np.mean(y_pred)
            std_actual = np.std(y_true)
            std_predicted = np.std(y_pred)

            # Calculate relative metrics (as percentage of mean)
            mae_percentage = (mae / mean_actual * 100) if mean_actual != 0 else np.nan
            rmse_percentage = (rmse / mean_actual * 100) if mean_actual != 0 else np.nan
            bias_percentage = (bias / mean_actual * 100) if mean_actual != 0 else np.nan

            metrics = {
                # Primary metrics
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MAPE': float(mape) if not np.isnan(mape) else None,
                'R2': float(r2) if not np.isnan(r2) else None,
                'Bias': float(bias),

                # Relative metrics (as % of mean)
                'MAE_percent': float(mae_percentage) if not np.isnan(mae_percentage) else None,
                'RMSE_percent': float(rmse_percentage) if not np.isnan(rmse_percentage) else None,
                'Bias_percent': float(bias_percentage) if not np.isnan(bias_percentage) else None,

                # Descriptive statistics
                'mean_actual': float(mean_actual),
                'mean_predicted': float(mean_predicted),
                'std_actual': float(std_actual),
                'std_predicted': float(std_predicted),
                'n_samples': int(len(y_true)),

                # Interpretation helpers
                'bias_direction': 'over-forecast' if bias > 0 else 'under-forecast' if bias < 0 else 'neutral',
                'r2_quality': self._interpret_r2(r2) if not np.isnan(r2) else 'undefined'
            }

            return {
                "success": True,
                "metrics": metrics,
                "message": f"Calculated metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R\u00b2={r2:.3f}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error calculating metrics: {str(e)}",
                "metrics": {}
            }

    def _interpret_r2(self, r2: float) -> str:
        """Interpret R² score quality."""
        if r2 >= 0.9:
            return 'excellent'
        elif r2 >= 0.7:
            return 'good'
        elif r2 >= 0.5:
            return 'moderate'
        elif r2 >= 0:
            return 'poor'
        else:
            return 'very_poor'
