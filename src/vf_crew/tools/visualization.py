"""Visualization Tool for creating forecast charts."""

from typing import Type, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64


class VisualizationInput(BaseModel):
    """Input schema for VisualizationTool."""
    historical_data_file: str = Field(..., description="Path to pickled file with historical time series data")
    forecast_data_file: str = Field(..., description="Path to pickled file with forecast data")
    time_series_key: str = Field(..., description="Key of the time series to visualize")
    title: str = Field(..., description="Chart title")
    save_path: Optional[str] = Field(None, description="Path to save the figure")


class VisualizationTool(BaseTool):
    name: str = "Visualization Tool"
    description: str = (
        "Creates professional forecast visualization charts. "
        "Shows historical actuals, forecast predictions, and confidence intervals. "
        "Takes historical_data_file and forecast_data_file paths and creates chart for specified time_series_key. "
        "Can save to file or return as base64 encoded image."
    )
    args_schema: Type[BaseModel] = VisualizationInput

    def _run(
        self,
        historical_data_file: str,
        forecast_data_file: str,
        time_series_key: str,
        title: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create forecast visualization for a specific time series.

        Args:
            historical_data_file: Path to pickled file with historical data
            forecast_data_file: Path to pickled file with forecast data
            time_series_key: Which time series to visualize
            title: Chart title
            save_path: Optional path to save the figure

        Returns:
            Dictionary with visualization info and optionally base64 encoded image
        """
        try:
            import pickle

            # Load historical and forecast data from pickle files
            with open(historical_data_file, 'rb') as f:
                historical_groups = pickle.load(f)

            with open(forecast_data_file, 'rb') as f:
                forecast_groups = pickle.load(f)

            # Check if time series exists in both files
            if time_series_key not in historical_groups:
                return {
                    "success": False,
                    "error": f"Time series key '{time_series_key}' not found in historical data",
                    "image_base64": None
                }

            if time_series_key not in forecast_groups:
                return {
                    "success": False,
                    "error": f"Time series key '{time_series_key}' not found in forecast data",
                    "image_base64": None
                }

            # Extract specific time series data
            hist_data_dict = historical_groups[time_series_key]
            hist_df = hist_data_dict['data'] if isinstance(hist_data_dict, dict) and 'data' in hist_data_dict else hist_data_dict

            forecast_dict = forecast_groups[time_series_key]

            # Extract forecast components based on Prophet format
            if 'forecast_full' in forecast_dict:
                forecast_df = forecast_dict['forecast_full']
            else:
                return {
                    "success": False,
                    "error": "Unexpected forecast data format",
                    "image_base64": None
                }

            # Ensure dates are datetime
            hist_df['ds'] = pd.to_datetime(hist_df['ds'])
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

            # Separate historical fit and future forecast
            last_hist_date = hist_df['ds'].max()
            forecast_future = forecast_df[forecast_df['ds'] > last_hist_date].copy()
            forecast_hist = forecast_df[forecast_df['ds'] <= last_hist_date].copy()

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))

            # Plot historical actuals
            ax.plot(
                hist_df['ds'],
                hist_df['y'],
                'ko-',
                label='Actual',
                linewidth=1.5,
                markersize=4
            )

            # Plot historical fit (Prophet's fit on training data)
            if len(forecast_hist) > 0:
                ax.plot(
                    forecast_hist['ds'],
                    forecast_hist['yhat'],
                    'g--',
                    label='Model Fit',
                    linewidth=1.5,
                    alpha=0.7
                )

            # Plot future forecast
            if len(forecast_future) > 0:
                ax.plot(
                    forecast_future['ds'],
                    forecast_future['yhat'],
                    'b-',
                    label='Forecast',
                    linewidth=2
                )

                # Plot confidence interval for future
                ax.fill_between(
                    forecast_future['ds'],
                    forecast_future['yhat_lower'],
                    forecast_future['yhat_upper'],
                    alpha=0.2,
                    color='blue',
                    label='Confidence Interval (80%)'
                )

            # Formatting
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Volume', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45, ha='right')

            # Tight layout
            plt.tight_layout()

            # Save or encode
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                return {
                    "success": True,
                    "message": f"Chart saved to {save_path}",
                    "save_path": save_path
                }
            else:
                # Convert to base64 for embedding
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()

                return {
                    "success": True,
                    "message": "Chart created successfully",
                    "image_base64": image_base64,
                    "format": "png"
                }

        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return {
                "success": False,
                "error": f"Visualization failed: {str(e)}",
                "image_base64": None
            }
