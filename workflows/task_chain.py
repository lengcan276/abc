# workflows/task_chain.py
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI
import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any

try:
    from config import EXCITATION_SETTINGS
except ImportError:
    EXCITATION_SETTINGS = {}
class TaskChain:
    """
    Coordinates the execution of agents in a workflow using LangChain.
    """
    
    def __init__(self):
        """Initialize the TaskChain."""
        self.logger = logging.getLogger('TaskChain')
        self.initialize_agents()
        self.molecules = []  # Add molecules list
        
    def initialize_agents(self):
        """Initialize all agents."""
        from agents.data_agent import DataAgent
        from agents.feature_agent import FeatureAgent
        from agents.exploration_agent import ExplorationAgent
        from agents.model_agent import ModelAgent
        from agents.insight_agent import InsightAgent
        from utils.excitation_analyzer import ExcitationAnalyzer  # Add new analyzer
        
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.exploration_agent = ExplorationAgent()
        self.model_agent = ModelAgent()
        self.insight_agent = InsightAgent()
        self.excitation_analyzer = ExcitationAnalyzer()  # Initialize analyzer
        
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
            ),
            Tool(
                name="ReversedGapAnalysis",
                func=self.process_reversed_gap_analysis,
                description="Analyze reversed singlet-triplet gaps using TD-DFT calculations"
            )
        ]
        
        return tools
        
    def execute_data_extraction(self, base_dir):
        """Execute data extraction with the data agent."""
        try:
            self.data_agent.base_dir = base_dir
            result = self.data_agent.process_molecules()
            # Extract molecule names for later use
            if result and isinstance(result, str) and os.path.exists(result):
                df = pd.read_csv(result)
                self.molecules = df['Molecule'].unique().tolist()
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
            
    def process_reversed_gap_analysis(self, log_dir=None):
        """
        Process reversed gap analysis for excited state calculations.
        
        Args:
            log_dir: Directory containing Gaussian log files for excited state calculations
            
        Returns:
            dict: Analysis results including excitation data, candidates, and reports
        """
        print("=" * 60)
        print("Reversed Singlet-Triplet Gap Analysis")
        print("Based on wB97X-D/def2-TZVP method")
        print("=" * 60)
        
        if not log_dir:
            log_dir = os.path.join(self.data_agent.base_dir, 'excited_states')
        
        # 1. Extract all excited state data
        print("\n1. Extracting excited state data...")
        excitation_data = []
        
        # Get molecules list if not already loaded
        if not self.molecules:
            # Try to get from existing data
            extracted_dir = 'data/extracted'
            if os.path.exists(extracted_dir):
                for file in os.listdir(extracted_dir):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(extracted_dir, file))
                        if 'Molecule' in df.columns:
                            self.molecules = df['Molecule'].unique().tolist()
                            break
           # Get file patterns from config
        file_patterns = EXCITATION_SETTINGS.get('file_patterns', {})
        singlet_patterns = file_patterns.get('singlet', [
            '{molecule}_s0_singlet_wb97xd.log',
            '{molecule}/excited.log'
        ])
        triplet_patterns = file_patterns.get('triplet', [
            '{molecule}_s0_triplet_wb97xd.log',
            '{molecule}/triplet.log'
        ])
        
        for molecule in self.molecules:
            # Try each pattern for singlet files
            singlet_log = None
            for pattern in singlet_patterns:
                potential_path = os.path.join(log_dir, pattern.format(molecule=molecule))
                if os.path.exists(potential_path):
                    singlet_log = potential_path
                    break
            
            # Try each pattern for triplet files
            triplet_log = None
            for pattern in triplet_patterns:
                potential_path = os.path.join(log_dir, pattern.format(molecule=molecule))
                if os.path.exists(potential_path):
                    triplet_log = potential_path
                    break
            
            if singlet_log and triplet_log:
                try:
                    states = self.data_agent.extract_all_excited_states_combined(
                        singlet_log, triplet_log
                    )
                    
                    if states:
                        states['molecule'] = molecule
                        excitation_data.append(states)
                except Exception as e:
                    self.logger.warning(f"Failed to extract states for {molecule}: {e}")
            else:
                self.logger.warning(f"Missing log files for {molecule}")
        
        print(f"Successfully extracted excited state data for {len(excitation_data)} molecules")
        
        if not excitation_data:
            self.logger.error("No excitation data extracted. Check log file paths.")
            return None
        
        # 2. Analyze reversed gaps
        print("\n2. Analyzing reversed gaps...")
        analyzed_molecules = []
        
        for mol_states in excitation_data:
            try:
                analysis = self.excitation_analyzer.analyze_molecule_excitations(
                    mol_states, 
                    mol_states['molecule']
                )
                analyzed_molecules.append(analysis)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {mol_states['molecule']}: {e}")
        
        # 3. Generate comparison data
        print("\n3. Generating analysis report...")
        comparison_df = self.excitation_analyzer.generate_benchmark_comparison(analyzed_molecules)
        
        # Statistics
        print("\nDiscovered reversed gap types:")
        print(comparison_df['gap_type'].value_counts())
        
        # Use screening criteria from config
        criteria = EXCITATION_SETTINGS.get('screening_criteria', {})
        confidence_threshold = criteria.get('confidence_threshold', 0.7)
        gap_threshold = criteria.get('gap_threshold', -0.005)

        # High confidence reversals
        high_confidence = comparison_df[comparison_df['confidence'] > confidence_threshold]
        print(f"\nHigh confidence reversed gaps: {len(high_confidence)}")
        
        # 4. Save results
        output_settings = EXCITATION_SETTINGS.get('output_settings', {})
        output_dir = "data/reports/excitation_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed data in multiple formats if configured
        export_formats = output_settings.get('export_format', ['csv'])
        
        if 'csv' in export_formats:
            comparison_df.to_csv(
                os.path.join(output_dir, 'inverted_gaps_analysis.csv'), 
                index=False
            )
        
        if 'json' in export_formats:
            comparison_df.to_json(
                os.path.join(output_dir, 'inverted_gaps_analysis.json'),
                orient='records',
                indent=2
            )
        
        if 'xlsx' in export_formats:
            comparison_df.to_excel(
                os.path.join(output_dir, 'inverted_gaps_analysis.xlsx'),
                index=False
            )
        
        # Generate visualizations if configured
        if output_settings.get('plot_energy_diagrams', True):
            try:
                self.excitation_analyzer.plot_analysis_results(comparison_df, output_dir)
            except Exception as e:
                self.logger.warning(f"Failed to generate plots: {e}")
        
        # 5. Screen high quality candidates
        print("\n4. Screening high quality reversed TADF candidates...")
        candidates = comparison_df[
            (comparison_df['confidence'] > confidence_threshold) & 
            (comparison_df['calculated_gap'] < gap_threshold)
        ]
        
        print(f"Screened {len(candidates)} high quality candidates")
        
        # Save candidates separately
        if not candidates.empty:
            candidates.to_csv(
                os.path.join(output_dir, 'high_quality_candidates.csv'),
                index=False
            )
        
        # 6. Update feature engineering with new data
        print("\n5. Updating feature data...")
        try:
            self.feature_agent.update_with_excitation_analysis(comparison_df)
        except Exception as e:
            self.logger.warning(f"Failed to update features: {e}")
        
        return {
            'excitation_data': excitation_data,
            'analyzed_molecules': analyzed_molecules,
            'comparison_df': comparison_df,
            'candidates': candidates,
            'output_dir': output_dir
        }
    
    def run_complete_pipeline(self, base_dir, include_excitation_analysis=False):
        """
        Run the complete analysis pipeline.
        
        Args:
            base_dir: Base directory containing molecular data
            include_excitation_analysis: Whether to include reversed gap analysis
        """
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
            
            # 5. Optional: Reversed gap analysis
            excitation_result = None
            if include_excitation_analysis:
                self.logger.info("Starting reversed gap analysis...")
                excitation_result = self.process_reversed_gap_analysis()
            
            # 6. Insight generation
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
                'insights': insight_result,
                'excitation_analysis': excitation_result
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution error: {e}")
            return None