# workflows/task_chain.py
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI
import os
import logging

class TaskChain:
    """
    Coordinates the execution of agents in a workflow using LangChain.
    """
    
    def __init__(self):
        """Initialize the TaskChain."""
        self.logger = logging.getLogger('TaskChain')
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all agents."""
        from agents.data_agent import DataAgent
        from agents.feature_agent import FeatureAgent
        from agents.exploration_agent import ExplorationAgent
        from agents.model_agent import ModelAgent
        from agents.insight_agent import InsightAgent
        
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.exploration_agent = ExplorationAgent()
        self.model_agent = ModelAgent()
        self.insight_agent = InsightAgent()
        
    def create_agent_tools(self):
        """Create LangChain tools for each agent."""
        tools = [
            Tool(
                name="DataExtraction",
                func=self.execute_data_extraction,
                description="Extract molecular properties from Gaussian and CREST calculations"
            ),
            Tool(
                name="FeatureEngineering",
                func=self.execute_feature_engineering,
                description="Generate and process molecular descriptors and features"
            ),
            Tool(
                name="ExplorationAnalysis",
                func=self.execute_exploration_analysis,
                description="Analyze S1-T1 gap properties and identify reverse TADF candidates"
            ),
            Tool(
                name="PredictiveModeling",
                func=self.execute_predictive_modeling,
                description="Build and evaluate classification and regression models"
            ),
            Tool(
                name="InsightGeneration",
                func=self.execute_insight_generation,
                description="Generate quantum chemistry insights and design principles"
            )
        ]
        
        return tools
        
    def execute_data_extraction(self, base_dir):
        """Execute data extraction with the data agent."""
        try:
            self.data_agent.base_dir = base_dir
            result = self.data_agent.process_molecules()
            return result
        except Exception as e:
            self.logger.error(f"Data extraction error: {e}")
            return None
            
    def execute_feature_engineering(self, data_file):
        """Execute feature engineering with the feature agent."""
        try:
            result = self.feature_agent.run_feature_pipeline(data_file)
            return result
        except Exception as e:
            self.logger.error(f"Feature engineering error: {e}")
            return None
            
    def execute_exploration_analysis(self, gap_data):
        """Execute exploration analysis with the exploration agent."""
        try:
            neg_file = gap_data.get('negative_file')
            pos_file = gap_data.get('positive_file')
            
            result = self.exploration_agent.run_exploration_pipeline(neg_file, pos_file)
            return result
        except Exception as e:
            self.logger.error(f"Exploration analysis error: {e}")
            return None
            
    def execute_predictive_modeling(self, feature_file):
        """Execute predictive modeling with the model agent."""
        try:
            result = self.model_agent.run_modeling_pipeline(feature_file)
            return result
        except Exception as e:
            self.logger.error(f"Predictive modeling error: {e}")
            return None
            
    def execute_insight_generation(self, input_data):
        """Execute insight generation with the insight agent."""
        try:
            modeling_results = input_data.get('modeling_results')
            exploration_results = input_data.get('exploration_results')
            
            result = self.insight_agent.run_insight_pipeline(modeling_results, exploration_results)
            return result
        except Exception as e:
            self.logger.error(f"Insight generation error: {e}")
            return None
            
    def run_complete_pipeline(self, base_dir):
        """Run the complete analysis pipeline."""
        try:
            # 1. Extract data
            self.logger.info("Starting data extraction...")
            data_result = self.execute_data_extraction(base_dir)
            
            if not data_result:
                self.logger.error("Data extraction failed.")
                return None
                
            # 2. Feature engineering
            self.logger.info("Starting feature engineering...")
            feature_result = self.execute_feature_engineering(data_result)
            
            if not feature_result or 'feature_file' not in feature_result:
                self.logger.error("Feature engineering failed.")
                return None
                
            # 3. Exploration analysis
            self.logger.info("Starting exploration analysis...")
            exploration_result = self.execute_exploration_analysis(feature_result.get('gap_data'))
            
            # 4. Predictive modeling
            self.logger.info("Starting predictive modeling...")
            modeling_result = self.execute_predictive_modeling(feature_result.get('feature_file'))
            
            # 5. Insight generation
            self.logger.info("Starting insight generation...")
            insight_result = self.execute_insight_generation({
                'modeling_results': modeling_result,
                'exploration_results': exploration_result
            })
            
            return {
                'data': data_result,
                'features': feature_result,
                'exploration': exploration_result,
                'modeling': modeling_result,
                'insights': insight_result
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution error: {e}")
            return None
