import os
import subprocess
import logging
from pathlib import Path

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/annotator_subsampling_sweep.log'),
        logging.StreamHandler()
    ]
)

def run_sweep():
    """Run experiments with different numbers of annotators"""
    # Define the approaches to test
    approaches = ['aart', 'multitask', 'annotator_embedding']
    
    # Define the numbers of annotators to test
    # For HSB, we have 6 annotators total, so we'll test with 2-6 annotators
    n_annotators_list = [2, 3, 4, 5, 6]
    
    # Run experiments for each number of annotators
    for n_annotators in n_annotators_list:
        logging.info(f"\nRunning experiments with {n_annotators} annotators")
        
        # Construct command
        cmd = [
            'python', 'scripts/run_annotator_subsampling.py',
            '--approaches', *approaches,
            '--n_annotators', str(n_annotators)
        ]
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running experiment with {n_annotators} annotators: {str(e)}")
            continue

if __name__ == "__main__":
    run_sweep() 