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
    data_file: str = Field(..., description="Path to pickled Prophet-formatted time series data file")
    forecast_periods: int = Field(default=26, description="Number of periods (weeks) to forecast ahead")
    use_default_params: bool = Field(default=True, description="Use Prophet default parameters")


class ProphetModelTool(BaseTool):
    name: str = "Prophet Model Tool"
    description: str = (
        "Trains Facebook Prophet model on time series data and generates forecasts. "
        "Takes a data_file path from TimeSeriesPreparationTool and returns forecast_file path. "
        "Handles model training, prediction, and saves forecasts with confidence intervals."
    )
    args_schema: Type[BaseModel] = ProphetModelInput

    def _run(
        self,
        data_file: str,
        forecast_periods: int = 26,
        use_default_params: bool = True
    ) -> Dict[str, Any]:
        """
        Train Prophet model and generate forecasts.

        Args:
            data_file: Path to pickled Prophet-formatted time series data file
            forecast_periods: Number of periods to forecast (default: 26 weeks = 6 months)
            use_default_params: Use default Prophet parameters

        Returns:
            Dictionary containing:
                - forecast_file: Path to pickled forecast data
                - model_info: Model training information
                - success: Whether training was successful
        """
        try:
            import pickle
            import tempfile

            # Load data from pickle file
            with open(data_file, 'rb') as f:
                time_series_groups = pickle.load(f)

            # Process each time series and generate forecasts
            all_forecasts = {}
            forecast_summaries = []

            for ts_key, ts_data in time_series_groups.items():
                # Extract the DataFrame
                if isinstance(ts_data, dict) and 'data' in ts_data:
                    df = ts_data['data']
                    metadata = ts_data.get('metadata', {})
                else:
                    df = ts_data
                    metadata = {}

                # Validate data format
                if not isinstance(df, pd.DataFrame):
                    forecast_summaries.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Invalid data format'
                    })
                    continue

                if 'ds' not in df.columns or 'y' not in df.columns:
                    forecast_summaries.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Missing ds or y columns'
                    })
                    continue

                # Check minimum data points
                if len(df) < 10:
                    forecast_summaries.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': f'Insufficient data: {len(df)} points'
                    })
                    continue

                # Initialize Prophet model
                if use_default_params:
                    model = Prophet(
                        yearly_seasonality='auto',
                        weekly_seasonality='auto',
                        daily_seasonality=False,
                        seasonality_mode='multiplicative'
                    )
                else:
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

                # Store forecast with metadata
                all_forecasts[ts_key] = {
                    'forecast_full': forecast_output,
                    'historical_fit': historical_fit,
                    'future_forecast': future_forecast,
                    'metadata': metadata,
                    'model_info': {
                        'training_samples': len(df),
                        'forecast_periods': forecast_periods,
                        'training_start': df['ds'].min().strftime('%Y-%m-%d'),
                        'training_end': df['ds'].max().strftime('%Y-%m-%d'),
                        'forecast_start': future_forecast['ds'].min().strftime('%Y-%m-%d') if len(future_forecast) > 0 else 'N/A',
                        'forecast_end': future_forecast['ds'].max().strftime('%Y-%m-%d') if len(future_forecast) > 0 else 'N/A',
                        'parameters_used': 'default' if use_default_params else 'custom'
                    }
                }

                forecast_summaries.append({
                    'ts_key': ts_key,
                    'sku_name': metadata.get('Consumer Product', 'Unknown'),
                    'status': 'success',
                    'forecast_periods': forecast_periods,
                    'training_samples': len(df)
                })

            # Save all forecasts to temporary pickle file
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
            pickle.dump(all_forecasts, temp_file)
            temp_file.close()

            return {
                "success": True,
                "forecast_file": temp_file.name,
                "forecast_summaries": forecast_summaries[:10],  # First 10 for summary
                "total_forecasts": len(all_forecasts),
                "failed_forecasts": len([s for s in forecast_summaries if s['status'] == 'failed']),
                "message": f"Successfully generated forecasts for {len(all_forecasts)} time series (forecast_periods={forecast_periods}). Data saved to file."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Prophet model training failed: {str(e)}",
                "forecast_file": None,
                "total_forecasts": 0,
                "failed_forecasts": 0
            }
