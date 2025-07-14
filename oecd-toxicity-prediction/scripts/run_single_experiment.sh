#!/bin/bash

# =============================================================================
# Simple Single Experiment Runner for OECD Toxicity Prediction
# 
# This script runs a single experiment configuration for quick testing
# and development purposes.
#
# Usage:
#   ./run_single_experiment.sh --help
#   ./run_single_experiment.sh --type MF --model gbt --fingerprint Morgan --data TG201
#   ./run_single_experiment.sh --type MD --model xgb --fingerprint RDKit --data TG202
#
# =============================================================================

set -e

# Default values
EXPERIMENT_TYPE="MF"
MODEL="gbt"
FINGERPRINT="Morgan"
DATA_NAME="TG201"
VERBOSE=false

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

show_help() {
    cat << EOF
Simple Single Experiment Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --type TYPE          Experiment type: MF, MD, or COMBINED (default: MF)
    --model MODEL        Model to use: gbt, rf, xgb, dt, logistic (default: gbt)
    --fingerprint FP     Fingerprint type: Morgan, MACCS, RDKit, Layered (default: Morgan)
    --data DATA          Dataset name: TG201, TG202, etc. (default: TG201)
    --verbose            Enable verbose output
    --help               Show this help

EXAMPLES:
    # Run molecular fingerprints experiment
    $0 --type MF --model gbt --fingerprint Morgan --data TG201

    # Run molecular descriptors experiment  
    $0 --type MD --model xgb --fingerprint RDKit --data TG202

    # Run combined features experiment
    $0 --type COMBINED --model rf --fingerprint MACCS --data TG414

AVAILABLE OPTIONS:
    Types:        MF, MD, COMBINED
    Models:       gbt, rf, xgb, dt, logistic
    Fingerprints: Morgan, MACCS, RDKit, Layered
    Datasets:     TG201, TG202, TG203, TG210, TG211, TG414, TG416, etc.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                EXPERIMENT_TYPE="$2"
                shift 2
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --fingerprint)
                FINGERPRINT="$2"
                shift 2
                ;;
            --data)
                DATA_NAME="$2"
                shift 2
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

configure_experiment() {
    case $EXPERIMENT_TYPE in
        MF)
            FILE_PATH="data/raw/molecular_fingerprints/${DATA_NAME}.xlsx"
            MODEL_SUBDIR="molecular_fingerprints"
            SCRIPT_SUBDIR="molecular_fingerprints"
            USE_SCALER=false
            ;;
        MD)
            FILE_PATH="data/raw/molecular_descriptors/${DATA_NAME}_Descriptor_desalt.xlsx"
            MODEL_SUBDIR="molecular_descriptors"
            SCRIPT_SUBDIR="molecular_descriptors"
            USE_SCALER=true
            ;;
        COMBINED)
            FILE_PATH="data/raw/combined/${DATA_NAME}_Descriptor_desalt_250501.xlsx"
            MODEL_SUBDIR="combined"
            SCRIPT_SUBDIR="combined"
            USE_SCALER=true
            ;;
        *)
            log_error "Invalid experiment type: $EXPERIMENT_TYPE"
            exit 1
            ;;
    esac
    
    MODEL_SAVE_PATH="results/models/${MODEL_SUBDIR}/${DATA_NAME}"
    LOG_DIR="results/logs/${MODEL_SUBDIR}/${DATA_NAME}"
    SCRIPT_PATH="src/models/${SCRIPT_SUBDIR}/${MODEL}.py"
}

check_files() {
    # Check if data file exists (from project root)
    if [ ! -f "../$FILE_PATH" ]; then
        log_error "Data file not found: $FILE_PATH"
        log "Available data files:"
        if [ "$EXPERIMENT_TYPE" = "MF" ]; then
            find ../data -name "*.xlsx" -type f 2>/dev/null | head -5 || echo "  No .xlsx files found in data/"
        elif [ "$EXPERIMENT_TYPE" = "MD" ]; then
            find ../data -name "*_Descriptor_desalt.xlsx" -type f 2>/dev/null | head -5 || echo "  No descriptor files found"
        else
            find ../data -name "*_Descriptor_desalt_250501.xlsx" -type f 2>/dev/null | head -5 || echo "  No combined files found"
        fi
        exit 1
    fi
    
    # Check if script exists (from project root)
    if [ ! -f "../$SCRIPT_PATH" ]; then
        log_error "Script not found: $SCRIPT_PATH"
        log "Available scripts:"
        find ../src/models -name "*.py" -type f | grep -E "(gbt|rf|xgb|dt|logistic)\.py$" | head -5 || echo "  No model scripts found"
        exit 1
    fi
}

run_experiment() {
    # Create directories (from scripts directory perspective)
    mkdir -p "../$MODEL_SAVE_PATH"
    mkdir -p "../$LOG_DIR"
    
    # Build command - run from project root directory
    cmd=(
        python3 "src/models/${SCRIPT_SUBDIR}/${MODEL}.py"
        --fingerprint_type "$FINGERPRINT"
        --file_path "$FILE_PATH"
        --model_save_path "$MODEL_SAVE_PATH"
    )
    
    # Add scaler if needed
    if [ "$USE_SCALER" = true ]; then
        SCALER_SAVE_PATH="${MODEL_SAVE_PATH}/scaler_${FINGERPRINT}_${MODEL}.joblib"
        cmd+=(--scaler_save_path "$SCALER_SAVE_PATH")
    fi
    
    # Setup logging
    STDOUT_FILE="../${LOG_DIR}/${FINGERPRINT}_${MODEL}.log"
    STDERR_FILE="../${LOG_DIR}/${FINGERPRINT}_${MODEL}.err"
    
    log "Starting experiment: ${DATA_NAME}_${FINGERPRINT}_${MODEL}_${EXPERIMENT_TYPE}"
    log "Data file: $FILE_PATH"
    log "Output directory: $MODEL_SAVE_PATH"
    
    if [ "$VERBOSE" = true ]; then
        log "Command: ${cmd[*]}"
        log "Stdout: $STDOUT_FILE"
        log "Stderr: $STDERR_FILE"
    fi
    
    # Run the experiment from project root directory
    start_time=$(date +%s)
    experiment_id="${DATA_NAME}_${FINGERPRINT}_${MODEL}_${EXPERIMENT_TYPE}"
    
    # Change to project root and run the command with proper PYTHONPATH
    if (cd .. && PYTHONPATH="src:$PYTHONPATH" "${cmd[@]}") > "$STDOUT_FILE" 2> "$STDERR_FILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_success "Completed: $experiment_id (${duration}s)"
        
        # Show outputs
        if [ -f "$STDOUT_FILE" ]; then
            log "Output log: $STDOUT_FILE"
            if [ "$VERBOSE" = true ]; then
                echo "--- Last 10 lines of output ---"
                tail -10 "$STDOUT_FILE"
                echo "--- End of output ---"
            fi
        fi
        
        # Show model files
        log "Generated files:"
        find "$MODEL_SAVE_PATH" -name "*.joblib" -o -name "*.json" 2>/dev/null | while read -r file; do
            log "  $(basename "$file")"
        done
        
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_error "Experiment failed (${duration}s)"
        
        if [ -f "$STDERR_FILE" ] && [ -s "$STDERR_FILE" ]; then
            log_error "Error details:"
            cat "$STDERR_FILE"
        fi
        
        log_error "Full error log: $STDERR_FILE"
        return 1
    fi
}

main() {
    log "OECD Toxicity Prediction - Single Experiment Runner"
    log "================================================="
    
    parse_args "$@"
    configure_experiment
    check_files
    
    log "Configuration:"
    log "  Type: $EXPERIMENT_TYPE"
    log "  Model: $MODEL" 
    log "  Fingerprint: $FINGERPRINT"
    log "  Dataset: $DATA_NAME"
    log "  Use Scaler: $USE_SCALER"
    
    run_experiment
}

main "$@"
