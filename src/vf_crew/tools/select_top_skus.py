"""Select Top SKUs Tool for filtering time series by volume."""

from typing import Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SelectTopSKUsInput(BaseModel):
    """Input schema for SelectTopSKUsTool."""
    time_series_groups: Dict = Field(..., description="Dictionary of time series groups from ExcelParserTool")
    top_n: int = Field(default=10, description="Number of top SKUs to select by volume")


class SelectTopSKUsTool(BaseTool):
    name: str = "Select Top SKUs Tool"
    description: str = (
        "Selects top N SKUs (time series) by total volume. "
        "Filters the time series groups to keep only the highest volume products. "
        "Returns filtered time series groups and selection summary."
    )
    args_schema: Type[BaseModel] = SelectTopSKUsInput

    def _run(self, time_series_groups: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
        """
        Select top N SKUs by total volume.

        Args:
            time_series_groups: Dictionary of time series from ExcelParserTool
            top_n: Number of top SKUs to select (default: 10)

        Returns:
            Dictionary containing:
                - selected_series: Dictionary of top N time series
                - selection_summary: Summary of selection
                - skipped_count: Number of SKUs not selected
        """
        try:
            # Calculate total volume for each time series
            sku_volumes = []

            for ts_key, ts_data in time_series_groups.items():
                metadata = ts_data['metadata']
                total_volume = metadata.get('total_volume', 0)

                sku_volumes.append({
                    'ts_key': ts_key,
                    'total_volume': total_volume,
                    'metadata': metadata
                })

            # Sort by volume descending
            sku_volumes_sorted = sorted(sku_volumes, key=lambda x: x['total_volume'], reverse=True)

            # Select top N
            top_skus = sku_volumes_sorted[:top_n]

            # Create filtered time series groups
            selected_series = {}
            for sku in top_skus:
                ts_key = sku['ts_key']
                selected_series[ts_key] = time_series_groups[ts_key]

            # Create selection summary
            selection_summary = {
                'total_skus_available': len(time_series_groups),
                'top_n_requested': top_n,
                'top_n_selected': len(selected_series),
                'skipped_count': len(time_series_groups) - len(selected_series),
                'top_skus_info': [
                    {
                        'rank': idx + 1,
                        'sku_name': sku['metadata'].get('Consumer Product', 'Unknown'),
                        'total_volume': sku['total_volume'],
                        'liquid_class': sku['metadata'].get('RTD Liquid Class Filter', 'Unknown'),
                        'industry_type': sku['metadata'].get('Industry Type', 'Unknown'),
                        'partition': sku['metadata'].get('Partitions', 'Unknown'),
                        'n_weeks': sku['metadata'].get('n_weeks', 0)
                    }
                    for idx, sku in enumerate(top_skus)
                ]
            }

            return {
                "success": True,
                "selected_series": selected_series,
                "selection_summary": selection_summary,
                "skipped_count": len(time_series_groups) - len(selected_series),
                "message": f"Successfully selected top {len(selected_series)} SKUs by volume out of {len(time_series_groups)} total"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error selecting top SKUs: {str(e)}",
                "selected_series": {},
                "selection_summary": {},
                "skipped_count": 0
            }
