"""Prophet Model Tool for training and forecasting."""

from typing import Type, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class ProphetModelInput(BaseModel):
    """Input schema for ProphetModelTool."""
    time_series_data: Dict = Field(..., description="Prophet-formatted time series data (ds, y)")
    forecast_periods: int = Field(default=26, description="Number of periods (weeks) to forecast ahead")
    use_default_params: bool = Field(default=True, description="Use Prophet default parameters")


class ProphetModelTool(BaseTool):
    name: str = "Prophet Model Tool"
    description: str = (
        "Trains Facebook Prophet model on time series data and generates forecasts. "
        "Handles model training, prediction, and returns forecasts with confidence intervals. "
        "Uses default Prophet parameters for consistency."
    )
    args_schema: Type[BaseModel] = ProphetModelInput

    def _run(
        self,
        time_series_data: Dict[str, Any],
        forecast_periods: int = 26,
        use_default_params: bool = True
    ) -> Dict[str, Any]:
        """
        Train Prophet model and generate forecasts.

        Args:
            time_series_data: Dictionary with 'data' key containing Prophet-formatted DataFrame
            forecast_periods: Number of periods to forecast (default: 26 weeks = 6 months)
            use_default_params: Use default Prophet parameters

        Returns:
            Dictionary containing:
                - forecast: DataFrame with predictions
                - model_info: Model training information
                - success: Whether training was successful
        """
        try:
            # Extract the data
            if isinstance(time_series_data, dict) and 'data' in time_series_data:
                df = time_series_data['data']
            else:
                df = time_series_data

            # Validate data format
            if not isinstance(df, pd.DataFrame):
                return {
                    "success": False,
                    "error": "Input must be a pandas DataFrame",
                    "forecast": None
                }

            if 'ds' not in df.columns or 'y' not in df.columns:
                return {
                    "success": False,
                    "error": "DataFrame must have 'ds' and 'y' columns",
                    "forecast": None
                }

            # Check minimum data points
            if len(df) < 10:
                return {
                    "success": False,
                    "error": f"Insufficient data points: {len(df)} (minimum: 10)",
                    "forecast": None
                }

            # Initialize Prophet model with default params or custom
            if use_default_params:
                model = Prophet(
                    yearly_seasonality='auto',
                    weekly_seasonality='auto',
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
            else:
                # For future customization
                model = Prophet()

            # Fit the model
            model.fit(df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq='W')

            # Generate forecast
            forecast = model.predict(future)

            # Extract relevant columns
            forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

            # Add flag for historical vs forecast
            forecast_output['is_forecast'] = forecast_output['ds'] > df['ds'].max()

            # Separate historical fit and future forecast
            historical_fit = forecast_output[~forecast_output['is_forecast']].copy()
            future_forecast = forecast_output[forecast_output['is_forecast']].copy()

            # Model info
            model_info = {
                'training_samples': len(df),
                'forecast_periods': forecast_periods,
                'training_start': df['ds'].min().strftime('%Y-%m-%d'),
                'training_end': df['ds'].max().strftime('%Y-%m-%d'),
                'forecast_start': future_forecast['ds'].min().strftime('%Y-%m-%d'),
                'forecast_end': future_forecast['ds'].max().strftime('%Y-%m-%d'),
                'parameters_used': 'default' if use_default_params else 'custom'
            }

            return {
                "success": True,
                "forecast_full": forecast_output,
                "historical_fit": historical_fit,
                "future_forecast": future_forecast,
                "model_info": model_info,
                "message": f"Successfully trained Prophet model and generated {forecast_periods}-period forecast"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Prophet model training failed: {str(e)}",
                "forecast": None,
                "model_info": {}
            }
