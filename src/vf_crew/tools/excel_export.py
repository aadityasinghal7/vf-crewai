"""Excel Export Tool for writing results with visualizations."""

from typing import Type, Dict, Any, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import os
from datetime import datetime


class ExcelExportInput(BaseModel):
    """Input schema for ExcelExportTool."""
    forecasts_data: List[Dict] = Field(..., description="List of forecast results for all time series")
    validation_metrics: List[Dict] = Field(..., description="List of validation metrics for all time series")
    invalid_series: List[Dict] = Field(..., description="List of invalid/skipped time series")
    output_path: str = Field(..., description="Path to save the Excel file")
    summary_stats: Optional[Dict] = Field(None, description="Optional summary statistics")


class ExcelExportTool(BaseTool):
    name: str = "Excel Export Tool"
    description: str = (
        "Exports forecast results, validation metrics, and visualizations to Excel. "
        "Creates multiple sheets: forecasts, validation metrics, flagged series, and summary. "
        "Includes formatting and professional layout."
    )
    args_schema: Type[BaseModel] = ExcelExportInput

    def _run(
        self,
        forecasts_data: List[Dict],
        validation_metrics: List[Dict],
        invalid_series: List[Dict],
        output_path: str,
        summary_stats: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Export results to Excel file.

        Args:
            forecasts_data: List of forecast dictionaries
            validation_metrics: List of validation metric dictionaries
            invalid_series: List of invalid/skipped series
            output_path: Path to save Excel file
            summary_stats: Optional summary statistics

        Returns:
            Dictionary with export status and file info
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4472C4',
                    'font_color': 'white',
                    'border': 1
                })
                number_format = workbook.add_format({'num_format': '#,##0.00'})
                date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})

                # Sheet 1: Forecasts
                if forecasts_data:
                    forecasts_list = []
                    for item in forecasts_data:
                        # Extract metadata
                        metadata = item.get('metadata', {})
                        forecast_df = item.get('forecast', pd.DataFrame())

                        if not forecast_df.empty:
                            forecast_df = forecast_df.copy()
                            # Add metadata columns
                            forecast_df['Liquid_Class'] = metadata.get('RTD Liquid Class Filter', '')
                            forecast_df['Industry_Type'] = metadata.get('Industry Type', '')
                            forecast_df['Consumer_Product'] = metadata.get('Consumer Product', '')
                            forecast_df['Partition'] = metadata.get('Partitions', '')

                            forecasts_list.append(forecast_df)

                    if forecasts_list:
                        forecasts_combined = pd.concat(forecasts_list, ignore_index=True)
                        # Reorder columns
                        cols = ['Liquid_Class', 'Industry_Type', 'Consumer_Product', 'Partition',
                                'ds', 'yhat', 'yhat_lower', 'yhat_upper']
                        forecasts_combined = forecasts_combined[[c for c in cols if c in forecasts_combined.columns]]
                        forecasts_combined.to_excel(writer, sheet_name='Forecasts', index=False)

                        # Format the sheet
                        worksheet = writer.sheets['Forecasts']
                        for col_num, value in enumerate(forecasts_combined.columns.values):
                            worksheet.write(0, col_num, value, header_format)

                # Sheet 2: Validation Metrics
                if validation_metrics:
                    metrics_df = pd.DataFrame(validation_metrics)
                    metrics_df.to_excel(writer, sheet_name='Validation_Metrics', index=False)

                    worksheet = writer.sheets['Validation_Metrics']
                    for col_num, value in enumerate(metrics_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                # Sheet 3: Flagged/Skipped Series
                if invalid_series:
                    flagged_list = []
                    for item in invalid_series:
                        metadata = item.get('metadata', {})
                        issues = item.get('issues', [])
                        flagged_list.append({
                            'Liquid_Class': metadata.get('RTD Liquid Class Filter', ''),
                            'Industry_Type': metadata.get('Industry Type', ''),
                            'Consumer_Product': metadata.get('Consumer Product', ''),
                            'Partition': metadata.get('Partitions', ''),
                            'Issues': ' | '.join(issues) if isinstance(issues, list) else str(issues),
                            'N_Weeks': metadata.get('n_weeks', 0),
                            'Start_Date': metadata.get('start_date', ''),
                            'End_Date': metadata.get('end_date', '')
                        })

                    flagged_df = pd.DataFrame(flagged_list)
                    flagged_df.to_excel(writer, sheet_name='Flagged_Series', index=False)

                    worksheet = writer.sheets['Flagged_Series']
                    for col_num, value in enumerate(flagged_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column('E:E', 50)  # Widen Issues column

                # Sheet 4: Summary
                summary_data = []
                summary_data.append(['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                summary_data.append(['Total Forecasts Generated', len(forecasts_data)])
                summary_data.append(['Total Validation Metrics', len(validation_metrics)])
                summary_data.append(['Total Flagged Series', len(invalid_series)])

                if summary_stats:
                    summary_data.append(['', ''])
                    summary_data.append(['=== Summary Statistics ===', ''])
                    for key, value in summary_stats.items():
                        summary_data.append([str(key), str(value)])

                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                worksheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 40)

            return {
                "success": True,
                "output_path": output_path,
                "sheets_created": ['Forecasts', 'Validation_Metrics', 'Flagged_Series', 'Summary'],
                "forecasts_count": len(forecasts_data),
                "metrics_count": len(validation_metrics),
                "flagged_count": len(invalid_series),
                "message": f"Successfully exported results to {output_path}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Excel export failed: {str(e)}",
                "output_path": None
            }
