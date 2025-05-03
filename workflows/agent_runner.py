# agent_runner.py
from workflows.task_chain import TaskChain
import argparse
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reverse_tadf_runner.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('AgentRunner')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Reverse TADF Analysis System')
    
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing molecular calculation data')
    parser.add_argument('--data_file', type=str, default=None,
                       help='CSV file with extracted molecular data')
    parser.add_argument('--feature_file', type=str, default=None,
                       help='CSV file with processed features')
    parser.add_argument('--step', type=str,
                       choices=['data', 'features', 'exploration', 'modeling', 'insights', 'all'],
                       default='all',
                       help='Which step to run')
    
    return parser.parse_args()

def main():
    """Main entry point for command-line execution."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/extracted', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/reports', exist_ok=True)
    
    # Initialize task chain
    task_chain = TaskChain()
    
    if args.step == 'data' or args.step == 'all':
        if args.data_dir:
            logger.info(f"Running data extraction from {args.data_dir}...")
            data_result = task_chain.execute_data_extraction(args.data_dir)
            
            if data_result:
                logger.info(f"Data extraction completed. Results saved to {data_result}")
            else:
                logger.error("Data extraction failed.")
                if args.step == 'all':
                    logger.error("Aborting remaining steps.")
                    return
        else:
            logger.error("No data directory provided for extraction.")
            if args.step == 'all':
                logger.error("Aborting remaining steps.")
                return
                
    if args.step == 'features' or args.step == 'all':
        data_file = args.data_file if args.data_file else (data_result if 'data_result' in locals() else None)
        
        if data_file:
            logger.info(f"Running feature engineering on {data_file}...")
            feature_result = task_chain.execute_feature_engineering(data_file)
            
            if feature_result and 'feature_file' in feature_result:
                logger.info(f"Feature engineering completed. Results saved to {feature_result['feature_file']}")
            else:
                logger.error("Feature engineering failed.")
                if args.step == 'all':
                    logger.error("Aborting remaining steps.")
                    return
        else:
            logger.error("No data file provided for feature engineering.")
            if args.step == 'all':
                logger.error("Aborting remaining steps.")
                return
                
    if args.step == 'exploration' or args.step == 'all':
        gap_data = feature_result.get('gap_data') if 'feature_result' in locals() else None
        
        if gap_data:
            logger.info("Running exploration analysis...")
            exploration_result = task_chain.execute_exploration_analysis(gap_data)
            
            if exploration_result and 'report' in exploration_result:
                logger.info(f"Exploration analysis completed. Report saved to {exploration_result['report']}")
            else:
                logger.error("Exploration analysis failed.")
        else:
            logger.error("No gap data available for exploration analysis.")
            
    if args.step == 'modeling' or args.step == 'all':
        feature_file = args.feature_file if args.feature_file else (
            feature_result.get('feature_file') if 'feature_result' in locals() else None
        )
        
        if feature_file:
            logger.info(f"Running predictive modeling on {feature_file}...")
            modeling_result = task_chain.execute_predictive_modeling(feature_file)
            
            if modeling_result:
                logger.info("Predictive modeling completed.")
            else:
                logger.error("Predictive modeling failed.")
        else:
            logger.error("No feature file provided for predictive modeling.")
            
    if args.step == 'insights' or args.step == 'all':
        modeling_result = modeling_result if 'modeling_result' in locals() else None
        exploration_result = exploration_result if 'exploration_result' in locals() else None
        
        if modeling_result or exploration_result:
            logger.info("Generating insights report...")
            insight_result = task_chain.execute_insight_generation({
                'modeling_results': modeling_result,
                'exploration_results': exploration_result
            })
            
            if insight_result and 'report' in insight_result:
                logger.info(f"Insight generation completed. Report saved to {insight_result['report']}")
            else:
                logger.error("Insight generation failed.")
        else:
            logger.error("No modeling or exploration results available for insight generation.")
            
if __name__ == "__main__":
    main()
