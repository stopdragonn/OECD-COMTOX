#!/bin/bash

# =============================================================================
# Local Batch Experiment Runner for OECD Toxicity Prediction
# 
# This script runs batch experiments in standard computing environments
# without requiring SLURM or other cluster-specific tools.
#
# Features:
# - Supports MF, MD, and COMBINED experiments
# - Sequential and parallel execution modes
# - Progress tracking and logging
# - Error handling and resumption
# - Cross-platform compatibility
#
# Usage:
#   ./batch_experiment_local.sh --experiment_type MF --parallel 2
#   ./batch_experiment_local.sh --experiment_type MD --parallel 1
#   ./batch_experiment_local.sh --experiment_type COMBINED --sequential
#
# =============================================================================

set -e  # Exit on any error

# Default configuration
EXPERIMENT_TYPE="MF"
PARALLEL_JOBS=1
SEQUENTIAL_MODE=false
DRY_RUN=false
VERBOSE=false

# Model and fingerprint arrays
MODELS=("gbt")  # Add more: "rf" "logistic" "xgb" "dt"
FINGERPRINTS=("Morgan" "MACCS")  # Add more: "RDKit" "Layered"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Local Batch Experiment Runner for OECD Toxicity Prediction

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --experiment_type TYPE    Experiment type: MF, MD, or COMBINED (default: MF)
    --parallel N              Run N experiments in parallel (default: 1)
    --sequential              Run experiments sequentially (same as --parallel 1)
    --models MODEL1,MODEL2    Comma-separated list of models (default: gbt)
    --fingerprints FP1,FP2    Comma-separated list of fingerprints (default: Morgan,MACCS)
    --dry-run                 Show what would be executed without running
    --verbose                 Enable verbose output
    --help                    Show this help message

EXAMPLES:
    # Run MF experiments with 2 parallel jobs
    $0 --experiment_type MF --parallel 2

    # Run MD experiments sequentially
    $0 --experiment_type MD --sequential

    # Run with custom models and fingerprints
    $0 --experiment_type COMBINED --models gbt,rf,xgb --fingerprints Morgan,MACCS,RDKit

    # Dry run to see what would be executed
    $0 --experiment_type MF --dry-run

EXPERIMENT TYPES:
    MF        - Molecular Fingerprints (no scaler required)
    MD        - Molecular Descriptors (scaler required)
    COMBINED  - Combined MF+MD features (scaler required)
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --experiment_type)
                EXPERIMENT_TYPE="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --sequential)
                PARALLEL_JOBS=1
                SEQUENTIAL_MODE=true
                shift
                ;;
            --models)
                IFS=',' read -r -a MODELS <<< "$2"
                shift 2
                ;;
            --fingerprints)
                IFS=',' read -r -a FINGERPRINTS <<< "$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate experiment type
validate_experiment_type() {
    case $EXPERIMENT_TYPE in
        MF|MD|COMBINED)
            ;;
        *)
            log_error "Invalid experiment type: $EXPERIMENT_TYPE"
            log_error "Valid types: MF, MD, COMBINED"
            exit 1
            ;;
    esac
}

# Configure experiment settings
configure_experiment() {
    case $EXPERIMENT_TYPE in
        MF)
            DESCRIPTION="Molecular Fingerprints (no scaler required)"
            FILE_PATHS=(
                "../../data/raw/molecular_fingerprints/TG201.xlsx"
                # "../../data/raw/molecular_fingerprints/TG202.xlsx"
                # "../../data/raw/molecular_fingerprints/TG203.xlsx"
                # "../../data/raw/molecular_fingerprints/TG210.xlsx"  
                # "../../data/raw/molecular_fingerprints/TG211.xlsx"
            )
            MODEL_SUBDIR="molecular_fingerprints"
            SCRIPT_SUBDIR="molecular_fingerprints"
            USE_SCALER=false
            ;;
        MD)
            DESCRIPTION="Molecular Descriptors (scaler required)"
            FILE_PATHS=(
                "../../data/raw/molecular_descriptors/TG201_Descriptor_desalt.xlsx"
                # Add other MD files as needed
            )
            MODEL_SUBDIR="molecular_descriptors" 
            SCRIPT_SUBDIR="molecular_descriptors"
            USE_SCALER=true
            ;;
        COMBINED)
            DESCRIPTION="Combined MF+MD features (scaler required)"
            FILE_PATHS=(
                "../../data/raw/combined/TG414_Descriptor_desalt_250501.xlsx"
                # "../../data/raw/combined/TG416_Descriptor_desalt_250501.xlsx"
            )
            MODEL_SUBDIR="combined"
            SCRIPT_SUBDIR="combined"
            USE_SCALER=true
            ;;
    esac
    
    # Set paths
    BASE_MODEL_SAVE_PATH="../../results/models/${MODEL_SUBDIR}"
    BASE_LOG_PATH="../../results/logs/${MODEL_SUBDIR}"
    SCRIPT_BASE_PATH="../../src/models/${SCRIPT_SUBDIR}"
}

# Check system capabilities
check_system() {
    log "Checking system capabilities..."
    
    # Check if python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Please install Python 3.6+"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python version: $PYTHON_VERSION"
    
    # Check available CPU cores
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=1
    fi
    
    log "Available CPU cores: $CPU_CORES"
    
    # Recommend parallel jobs based on CPU cores
    if [ $PARALLEL_JOBS -gt $CPU_CORES ]; then
        log_warning "Parallel jobs ($PARALLEL_JOBS) exceeds CPU cores ($CPU_CORES)"
        log_warning "Consider reducing parallel jobs for better performance"
    fi
    
    # Check available memory (Linux only)
    if [ -f /proc/meminfo ]; then
        AVAILABLE_MEM=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
        log "Available memory: ${AVAILABLE_MEM} GB"
        
        # Recommend memory usage (assuming 2GB per job)
        RECOMMENDED_MEM=$(echo "$PARALLEL_JOBS * 2" | bc -l 2>/dev/null || echo "$((PARALLEL_JOBS * 2))")
        if (( $(echo "$RECOMMENDED_MEM > $AVAILABLE_MEM" | bc -l 2>/dev/null || [ $RECOMMENDED_MEM -gt ${AVAILABLE_MEM%.*} ]) )); then
            log_warning "Recommended memory (${RECOMMENDED_MEM}GB) may exceed available memory (${AVAILABLE_MEM}GB)"
        fi
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    for file_path in "${FILE_PATHS[@]}"; do
        target_name=$(basename "$file_path" .xlsx)
        model_save_path="${BASE_MODEL_SAVE_PATH}/${target_name}"
        log_dir="${BASE_LOG_PATH}/${target_name}"
        
        mkdir -p "$model_save_path"
        mkdir -p "$log_dir"
        
        if [ "$VERBOSE" = true ]; then
            log "Created: $model_save_path"
            log "Created: $log_dir"
        fi
    done
}

# Generate experiment list
generate_experiments() {
    EXPERIMENTS=()
    
    for file_path in "${FILE_PATHS[@]}"; do
        for model in "${MODELS[@]}"; do
            for fingerprint in "${FINGERPRINTS[@]}"; do
                target_name=$(basename "$file_path" .xlsx)
                experiment_id="${target_name}_${fingerprint}_${model}"
                EXPERIMENTS+=("$file_path|$model|$fingerprint|$experiment_id")
            done
        done
    done
}

# Run a single experiment
run_single_experiment() {
    local experiment_data="$1"
    IFS='|' read -r file_path model fingerprint experiment_id <<< "$experiment_data"
    
    target_name=$(basename "$file_path" .xlsx)
    model_save_path="${BASE_MODEL_SAVE_PATH}/${target_name}"
    log_dir="${BASE_LOG_PATH}/${target_name}"
    script_path="${SCRIPT_BASE_PATH}/${model}.py"
    
    # We're already in project root when script is run from project root
    # No need to change directory
    
    # Setup logging files with correct paths
    stdout_file="${log_dir#../../}/${fingerprint}_${model}.log"
    stderr_file="${log_dir#../../}/${fingerprint}_${model}.err"
    
    # Build command with correct paths from project root
    cmd=(
        python3 "src/models/${SCRIPT_SUBDIR}/${model}.py"
        --fingerprint_type "$fingerprint"
        --file_path "${file_path#../../}"
        --model_save_path "${model_save_path#../../}"
    )
    
    # Add scaler path if needed
    if [ "$USE_SCALER" = true ]; then
        scaler_save_path="${model_save_path}/scaler_${fingerprint}_${model}.joblib"
        cmd+=(--scaler_save_path "${scaler_save_path#../../}")
    fi
    
    # Ensure directories exist from project root
    mkdir -p "${model_save_path#../../}" "${log_dir#../../}"
    
    # Debug: print current directory and command
    if [ "$VERBOSE" = true ]; then
        log "Current directory: $(pwd)"
        log "Command array length: ${#cmd[@]}"
        log "Full command: ${cmd[*]}"
        log "Python path: $PYTHONPATH"
        log "Working directory check: $(ls -la | head -3)"
    fi

    # Set PYTHONPATH
    export PYTHONPATH="src:$PYTHONPATH"
    
    if [ "$VERBOSE" = true ]; then
        log "Running: ${cmd[*]}"
    fi
    
    # Run the experiment
    start_time=$(date +%s)
    
    if "${cmd[@]}" > "$stdout_file" 2> "$stderr_file"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_success "Completed: $experiment_id (${duration}s)"
        # No need to change back since we didn't change directory
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_error "Failed: $experiment_id (${duration}s)"
        log_error "Check error log: $stderr_file"
        return 1
    fi
}

# Run experiments sequentially
run_experiments_sequential() {
    log "Running experiments sequentially..."
    
    local success_count=0
    local total_count=${#EXPERIMENTS[@]}
    local start_time=$(date +%s)
    
    for i in "${!EXPERIMENTS[@]}"; do
        local progress=$(( (i + 1) * 100 / total_count ))
        log "Progress: $((i + 1))/$total_count ($progress%)"
        
        if run_single_experiment "${EXPERIMENTS[$i]}"; then
            ((success_count++))
        fi
    done
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    log_success "Sequential execution completed"
    log "Total time: ${total_duration}s ($(( total_duration / 60 ))m $(( total_duration % 60 ))s)"
    log "Success rate: $success_count/$total_count ($(( success_count * 100 / total_count ))%)"
}

# Run experiments in parallel (using background processes)
run_experiments_parallel() {
    log "Running experiments with $PARALLEL_JOBS parallel jobs..."
    
    local success_count=0
    local total_count=${#EXPERIMENTS[@]}
    local start_time=$(date +%s)
    local running_jobs=0
    local completed=0
    
    # Create temporary directory for job tracking
    local temp_dir=$(mktemp -d)
    local job_list=()
    
    for i in "${!EXPERIMENTS[@]}"; do
        # Wait if we have reached max parallel jobs
        while [ $running_jobs -ge $PARALLEL_JOBS ]; do
            # Check for completed jobs
            for j in "${!job_list[@]}"; do
                if [ -n "${job_list[$j]}" ]; then
                    local job_pid="${job_list[$j]}"
                    if ! kill -0 "$job_pid" 2>/dev/null; then
                        # Job completed
                        wait "$job_pid"
                        local job_status=$?
                        
                        if [ $job_status -eq 0 ]; then
                            ((success_count++))
                        fi
                        
                        ((completed++))
                        ((running_jobs--))
                        unset job_list[$j]
                        
                        local progress=$(( completed * 100 / total_count ))
                        log "Progress: $completed/$total_count ($progress%)"
                    fi
                fi
            done
            sleep 1
        done
        
        # Start new job
        run_single_experiment "${EXPERIMENTS[$i]}" &
        local new_job_pid=$!
        job_list+=("$new_job_pid")
        ((running_jobs++))
        
        if [ "$VERBOSE" = true ]; then
            IFS='|' read -r _ _ _ experiment_id <<< "${EXPERIMENTS[$i]}"
            log "Started job: $experiment_id (PID: $new_job_pid)"
        fi
    done
    
    # Wait for remaining jobs
    log "Waiting for remaining jobs to complete..."
    for job_pid in "${job_list[@]}"; do
        if [ -n "$job_pid" ]; then
            wait "$job_pid"
            local job_status=$?
            if [ $job_status -eq 0 ]; then
                ((success_count++))
            fi
            ((completed++))
            
            local progress=$(( completed * 100 / total_count ))
            log "Progress: $completed/$total_count ($progress%)"
        fi
    done
    
    # Cleanup
    rm -rf "$temp_dir"
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    log_success "Parallel execution completed"
    log "Total time: ${total_duration}s ($(( total_duration / 60 ))m $(( total_duration % 60 ))s)"
    log "Success rate: $success_count/$total_count ($(( success_count * 100 / total_count ))%)"
}

# Show dry run information
show_dry_run() {
    log "DRY RUN - Experiments that would be executed:"
    log "==========================================="
    log "Experiment Type: $EXPERIMENT_TYPE - $DESCRIPTION"
    log "Models: ${MODELS[*]}"
    log "Fingerprints: ${FINGERPRINTS[*]}"
    log "Parallel Jobs: $PARALLEL_JOBS"
    log "Use Scaler: $USE_SCALER"
    log "Total Experiments: ${#EXPERIMENTS[@]}"
    log ""
    
    for i in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r file_path model fingerprint experiment_id <<< "${EXPERIMENTS[$i]}"
        target_name=$(basename "$file_path" .xlsx)
        printf "%3d. %-20s + %-8s + %-10s\n" $((i+1)) "$target_name" "$fingerprint" "$model"
    done
    
    log ""
    log "To execute these experiments, run without --dry-run"
}

# Main function
main() {
    log "OECD Toxicity Prediction - Local Batch Experiment Runner"
    log "========================================================"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate and configure
    validate_experiment_type
    configure_experiment
    check_system
    
    # Generate experiments
    generate_experiments
    
    # Show configuration
    log "Configuration:"
    log "  Experiment Type: $EXPERIMENT_TYPE - $DESCRIPTION"
    log "  Models: ${MODELS[*]}"
    log "  Fingerprints: ${FINGERPRINTS[*]}"
    log "  Total Experiments: ${#EXPERIMENTS[@]}"
    log "  Parallel Jobs: $PARALLEL_JOBS"
    log "  Use Scaler: $USE_SCALER"
    
    # Handle dry run
    if [ "$DRY_RUN" = true ]; then
        show_dry_run
        return 0
    fi
    
    # Create directories
    create_directories
    
    # Run experiments
    if [ $PARALLEL_JOBS -eq 1 ]; then
        run_experiments_sequential
    else
        run_experiments_parallel
    fi
    
    log_success "Batch experiment completed!"
}

# Run main function with all arguments
main "$@"
