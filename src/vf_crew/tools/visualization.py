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
    historical_data: Any = Field(..., description="Historical data with 'ds' and 'y' columns")
    forecast_data: Any = Field(..., description="Forecast data with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'")
    title: str = Field(..., description="Chart title")
    save_path: Optional[str] = Field(None, description="Path to save the figure")


class VisualizationTool(BaseTool):
    name: str = "Visualization Tool"
    description: str = (
        "Creates professional forecast visualization charts. "
        "Shows historical actuals, forecast predictions, and confidence intervals. "
        "Can save to file or return as base64 encoded image."
    )
    args_schema: Type[BaseModel] = VisualizationInput

    def _run(
        self,
        historical_data: Any,
        forecast_data: Any,
        title: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create forecast visualization.

        Args:
            historical_data: DataFrame with historical values (ds, y)
            forecast_data: DataFrame with forecasts (ds, yhat, yhat_lower, yhat_upper)
            title: Chart title
            save_path: Optional path to save the figure

        Returns:
            Dictionary with visualization info and optionally base64 encoded image
        """
        try:
            # Convert to DataFrames if needed
            if isinstance(historical_data, dict):
                hist_df = pd.DataFrame(historical_data)
            else:
                hist_df = historical_data.copy()

            if isinstance(forecast_data, dict):
                forecast_df = pd.DataFrame(forecast_data)
            else:
                forecast_df = forecast_data.copy()

            # Ensure dates are datetime
            hist_df['ds'] = pd.to_datetime(hist_df['ds'])
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

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

            # Plot forecast
            ax.plot(
                forecast_df['ds'],
                forecast_df['yhat'],
                'b-',
                label='Forecast',
                linewidth=2
            )

            # Plot confidence interval
            ax.fill_between(
                forecast_df['ds'],
                forecast_df['yhat_lower'],
                forecast_df['yhat_upper'],
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
