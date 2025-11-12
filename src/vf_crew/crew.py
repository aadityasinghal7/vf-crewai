from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

# Import custom tools
from vf_crew.tools.excel_parser import ExcelParserTool
from vf_crew.tools.select_top_skus import SelectTopSKUsTool
from vf_crew.tools.data_validation import DataValidationTool
from vf_crew.tools.ts_preparation import TimeSeriesPreparationTool
from vf_crew.tools.prophet_model import ProphetModelTool
from vf_crew.tools.train_test_split import TrainTestSplitTool
from vf_crew.tools.metrics_calculator import MetricsCalculatorTool
from vf_crew.tools.visualization import VisualizationTool
from vf_crew.tools.excel_export import ExcelExportTool


@CrewBase
class VfCrew():
    """Volume Forecasting Crew using CrewAI and Facebook Prophet"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def _get_llm(self):
        """Get LLM instance configured for Anthropic Claude"""
        return LLM(
            model=f"anthropic/{os.getenv('MODEL', 'claude-3-5-sonnet-20241022')}",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    # Agent definitions with their respective tools
    @agent
    def data_management_agent(self) -> Agent:
        """Agent responsible for data loading, validation, and preparation"""
        return Agent(
            config=self.agents_config['data_management_agent'],
            tools=[
                ExcelParserTool(),
                SelectTopSKUsTool(),
                DataValidationTool(),
                TimeSeriesPreparationTool()
            ],
            llm=self._get_llm(),
            verbose=True
        )

    @agent
    def ml_forecasting_agent(self) -> Agent:
        """Agent responsible for training Prophet models and generating forecasts"""
        return Agent(
            config=self.agents_config['ml_forecasting_agent'],
            tools=[
                ProphetModelTool()
            ],
            llm=self._get_llm(),
            verbose=True
        )

    @agent
    def validation_agent(self) -> Agent:
        """Agent responsible for validation, metrics calculation, visualization, and export"""
        return Agent(
            config=self.agents_config['validation_agent'],
            tools=[
                TrainTestSplitTool(),
                ProphetModelTool(),
                MetricsCalculatorTool(),
                VisualizationTool(),
                ExcelExportTool()
            ],
            llm=self._get_llm(),
            verbose=True
        )

    # Task definitions
    @task
    def data_preparation_task(self) -> Task:
        """Task for data loading, validation, and preparation"""
        return Task(
            config=self.tasks_config['data_preparation_task'],
        )

    @task
    def validation_with_split_task(self) -> Task:
        """Task for performing validation using train-test split"""
        return Task(
            config=self.tasks_config['validation_with_split_task'],
        )

    @task
    def final_forecast_generation_task(self) -> Task:
        """Task for generating final forecasts using full dataset"""
        return Task(
            config=self.tasks_config['final_forecast_generation_task'],
        )

    @task
    def results_export_task(self) -> Task:
        """Task for exporting results to Excel with visualizations"""
        return Task(
            config=self.tasks_config['results_export_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Volume Forecasting Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
