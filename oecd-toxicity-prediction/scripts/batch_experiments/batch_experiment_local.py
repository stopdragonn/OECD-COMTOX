#!/usr/bin/env python3
"""
Local Batch Experiment Runner for OECD Toxicity Prediction

This script runs batch experiments in local/standard computing environments
without requiring SLURM or cluster-specific tools.

Features:
- Supports MF, MD, and COMBINED experiments
- Parallel processing using multiprocessing
- Progress tracking and logging
- Error handling and resumption
- Memory and CPU monitoring

Usage:
    python batch_experiment_local.py --experiment_type MF --max_workers 4
    python batch_experiment_local.py --experiment_type MD --max_workers 2
    python batch_experiment_local.py --experiment_type COMBINED --max_workers 2

Author: OECD Toxicity Prediction Team
"""

import argparse
import sys
import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Try to import psutil, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System resource monitoring will be limited.")

# Setup logging
def setup_logging(experiment_type):
    """Setup logging configuration"""
    log_dir = Path("../../results/logs/batch_experiments")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_{experiment_type}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Configuration for different experiment types
EXPERIMENT_CONFIGS = {
    "MF": {
        "description": "Molecular Fingerprints (no scaler required)",
        "file_paths": [
            "../data/raw/molecular_fingerprints/TG201.xlsx",
            # "../data/raw/molecular_fingerprints/TG202.xlsx",
            # "../data/raw/molecular_fingerprints/TG203.xlsx", 
            # "../data/raw/molecular_fingerprints/TG210.xlsx",
            # "../data/raw/molecular_fingerprints/TG211.xlsx"
        ],
        "model_subdir": "molecular_fingerprints",
        "script_subdir": "molecular_fingerprints",
        "use_scaler": False
    },
    "MD": {
        "description": "Molecular Descriptors (scaler required)", 
        "file_paths": [
            "../data/raw/molecular_descriptors/TG201_Descriptor_desalt.xlsx",
            # Add other MD files as needed
        ],
        "model_subdir": "molecular_descriptors",
        "script_subdir": "molecular_descriptors", 
        "use_scaler": True
    },
    "COMBINED": {
        "description": "Combined MF+MD features (scaler required)",
        "file_paths": [
            "../data/raw/combined/TG414_Descriptor_desalt_250501.xlsx",
            # "../data/raw/combined/TG416_Descriptor_desalt_250501.xlsx"
        ],
        "model_subdir": "combined",
        "script_subdir": "combined",
        "use_scaler": True
    }
}

# Model and fingerprint configurations
MODELS = ['gbt', 'rf', 'logistic', 'xgb', 'dt']
FINGERPRINTS = ["Morgan", "MACCS", "RDKit", "Layered"]

def check_system_resources():
    """Check and report system resources"""
    if PSUTIL_AVAILABLE:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(f"System Resources:")
        logger.info(f"  CPU cores: {cpu_count}")
        logger.info(f"  Memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
        logger.info(f"  Disk: {disk.free // (1024**3)} GB free")
        
        return cpu_count, memory.available
    else:
        # Fallback without psutil
        try:
            # Try to get CPU count from os module
            cpu_count = os.cpu_count() or 1
        except:
            cpu_count = 1
        
        logger.info(f"System Resources (limited info):")
        logger.info(f"  CPU cores: {cpu_count}")
        logger.info(f"  Memory: Information not available (psutil not installed)")
        
        # Return conservative estimates
        return cpu_count, 4 * 1024**3  # Assume 4GB available

def run_single_experiment(args_tuple):
    """Run a single experiment configuration"""
    model, fingerprint, file_path, config, base_paths = args_tuple
    
    target_name = Path(file_path).stem
    model_save_path = base_paths['model'] / target_name
    log_dir = base_paths['log'] / target_name
    
    # Create directories
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment command
    script_path = Path("../../src/models") / config['script_subdir'] / f"{model}.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--fingerprint_type", fingerprint,
        "--file_path", file_path,
        "--model_save_path", str(model_save_path)
    ]
    
    # Add scaler path if needed
    if config['use_scaler']:
        scaler_save_path = model_save_path / f"scaler_{fingerprint}_{model}.joblib"
        cmd.extend(["--scaler_save_path", str(scaler_save_path)])
    
    # Setup logging files
    stdout_file = log_dir / f"{fingerprint}_{model}.log"
    stderr_file = log_dir / f"{fingerprint}_{model}.err"
    
    start_time = time.time()
    experiment_id = f"{target_name}_{fingerprint}_{model}"
    
    try:
        logger.info(f"Starting experiment: {experiment_id}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run the experiment
        with open(stdout_file, 'w') as stdout, open(stderr_file, 'w') as stderr:
            result = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=3600  # 1 hour timeout
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ Completed: {experiment_id} ({duration:.1f}s)")
            return {
                'experiment_id': experiment_id,
                'status': 'success',
                'duration': duration,
                'stdout_file': str(stdout_file),
                'stderr_file': str(stderr_file)
            }
        else:
            logger.error(f"✗ Failed: {experiment_id} (exit code: {result.returncode})")
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'exit_code': result.returncode,
                'duration': duration,
                'stdout_file': str(stdout_file),
                'stderr_file': str(stderr_file)
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout: {experiment_id}")
        return {
            'experiment_id': experiment_id,
            'status': 'timeout',
            'duration': 3600,
            'stdout_file': str(stdout_file),
            'stderr_file': str(stderr_file)
        }
    except Exception as e:
        logger.error(f"✗ Error: {experiment_id} - {str(e)}")
        return {
            'experiment_id': experiment_id,
            'status': 'error',
            'error': str(e),
            'stdout_file': str(stdout_file),
            'stderr_file': str(stderr_file)
        }

def generate_experiments(config, models, fingerprints):
    """Generate all experiment combinations"""
    experiments = []
    base_paths = {
        'model': Path("../../results/models") / config['model_subdir'],
        'log': Path("../../results/logs") / config['model_subdir']
    }
    
    for file_path in config['file_paths']:
        for model in models:
            for fingerprint in fingerprints:
                experiments.append((model, fingerprint, file_path, config, base_paths))
    
    return experiments

def save_results(results, experiment_type):
    """Save experiment results to JSON file"""
    results_dir = Path("../../results/logs/batch_experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"batch_results_{experiment_type}_{timestamp}.json"
    
    summary = {
        'experiment_type': experiment_type,
        'timestamp': timestamp,
        'total_experiments': len(results),
        'successful': len([r for r in results if r['status'] == 'success']),
        'failed': len([r for r in results if r['status'] == 'failed']),
        'timeout': len([r for r in results if r['status'] == 'timeout']),
        'error': len([r for r in results if r['status'] == 'error']),
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Run batch experiments locally')
    parser.add_argument('--experiment_type', required=True, choices=['MF', 'MD', 'COMBINED'],
                        help='Type of experiment to run')
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Maximum number of parallel workers (default: 2)')
    parser.add_argument('--models', nargs='+', default=['gbt'],
                        choices=MODELS, help='Models to train')
    parser.add_argument('--fingerprints', nargs='+', default=['Morgan', 'MACCS'],
                        choices=FINGERPRINTS, help='Fingerprint types to use')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show experiments that would be run without executing')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.experiment_type)
    
    # Validate experiment type
    if args.experiment_type not in EXPERIMENT_CONFIGS:
        logger.error(f"Invalid experiment type: {args.experiment_type}")
        sys.exit(1)
    
    config = EXPERIMENT_CONFIGS[args.experiment_type]
    
    # Check system resources
    cpu_count, available_memory = check_system_resources()
    
    # Adjust max_workers based on system resources
    recommended_workers = min(args.max_workers, cpu_count, available_memory // (2 * 1024**3))  # 2GB per worker
    if recommended_workers < args.max_workers:
        logger.warning(f"Reducing workers from {args.max_workers} to {recommended_workers} based on system resources")
        args.max_workers = recommended_workers
    
    # Generate experiments
    experiments = generate_experiments(config, args.models, args.fingerprints)
    
    logger.info(f"Experiment Configuration:")
    logger.info(f"  Type: {args.experiment_type} - {config['description']}")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Fingerprints: {args.fingerprints}")
    logger.info(f"  Total experiments: {len(experiments)}")
    logger.info(f"  Max workers: {args.max_workers}")
    logger.info(f"  Scaler required: {config['use_scaler']}")
    
    if args.dry_run:
        logger.info("DRY RUN - Experiments that would be executed:")
        for i, (model, fingerprint, file_path, _, _) in enumerate(experiments, 1):
            target_name = Path(file_path).stem
            logger.info(f"  {i:2d}. {target_name} + {fingerprint} + {model}")
        return
    
    # Run experiments
    logger.info(f"Starting {len(experiments)} experiments...")
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all experiments
        future_to_experiment = {
            executor.submit(run_single_experiment, exp): exp 
            for exp in experiments
        }
        
        # Process completed experiments
        completed = 0
        for future in as_completed(future_to_experiment):
            result = future.result()
            results.append(result)
            completed += 1
            
            progress = (completed / len(experiments)) * 100
            logger.info(f"Progress: {completed}/{len(experiments)} ({progress:.1f}%)")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save and summarize results
    summary = save_results(results, args.experiment_type)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Experiment Type: {args.experiment_type}")
    logger.info(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Total Experiments: {summary['total_experiments']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Timeout: {summary['timeout']}")
    logger.info(f"Error: {summary['error']}")
    logger.info(f"Success Rate: {(summary['successful']/summary['total_experiments']*100):.1f}%")
    
    if summary['failed'] > 0 or summary['timeout'] > 0 or summary['error'] > 0:
        logger.info("\nFailed experiments:")
        for result in results:
            if result['status'] != 'success':
                logger.info(f"  {result['experiment_id']}: {result['status']}")
                if 'stderr_file' in result:
                    logger.info(f"    Error log: {result['stderr_file']}")

if __name__ == "__main__":
    main()
