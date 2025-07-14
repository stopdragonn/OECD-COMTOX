#!/bin/bash

# 실험 타입 설정 (MF: Molecular Fingerprints, MD: Molecular Descriptors, COMBINED: MF+MD)
EXPERIMENT_TYPE="MF"  # Change to "MD" or "COMBINED" as needed

# 모델 리스트 정의
models=('gbt') # 'rf' 'logistic' 'xgb'  'dt'
fingerprints=( "Morgan" "MACCS") #"RDKit" "Layered"

# 실험 타입별 파일 경로 설정
if [ "$EXPERIMENT_TYPE" == "MF" ]; then
    # Molecular Fingerprints 데이터 (스케일러 불필요)
    file_paths=('../../data/raw/molecular_fingerprints/TG201.xlsx') # Add other MF files as needed
    model_subdir="molecular_fingerprints"
    use_scaler=false
elif [ "$EXPERIMENT_TYPE" == "MD" ]; then
    # Molecular Descriptors 데이터 (스케일러 필요)
    file_paths=('../../data/raw/molecular_descriptors/TG201_Descriptor_desalt.xlsx') # Add other MD files
    model_subdir="molecular_descriptors"
    use_scaler=true
elif [ "$EXPERIMENT_TYPE" == "COMBINED" ]; then
    # Combined MF+MD 데이터 (스케일러 필요)
    file_paths=('../../data/raw/combined/TG414_Descriptor_desalt_250501.xlsx' '../../data/raw/combined/TG416_Descriptor_desalt_250501.xlsx')
    model_subdir="combined"
    use_scaler=true
fi

# 모델 저장 경로 및 로그 기본 경로
base_model_save_path="../../results/models/${model_subdir}"
base_log_path="../../results/logs/${model_subdir}"

# SLURM 최대 제출 작업 수 제한
max_jobs_per_user=10  # 20개의 작업을 동시에 제출할 수 있다고 가정

# 각 파일에 대해, 모델 및 지문 조합에 대한 실험 실행
for file_path in "${file_paths[@]}"; do
    # 타겟 파일 이름 추출
    target_name=$(basename "$file_path" .xlsx)

    # 타겟 파일에 맞는 모델 저장 경로 설정
    model_save_path="${base_model_save_path}/${target_name}"
    
    # 타겟 파일에 맞는 로그 디렉토리 설정
    log_dir="${base_log_path}/${target_name}"

    # 각 타겟에 대한 저장 경로 및 로그 디렉토리 생성
    mkdir -p "${model_save_path}"
    mkdir -p "${log_dir}"

    # 각 모델과 지문에 대해 작업 제출
    for model in "${models[@]}"; do
        for fingerprint in "${fingerprints[@]}"; do
            echo "Submitting job for target: $file_path, model: $model, fingerprint: $fingerprint, experiment_type: $EXPERIMENT_TYPE"

            # 제출된 작업 수 확인
            current_jobs=$(squeue -u $USER | wc -l)

            # 제출된 작업 수가 최대값에 도달하면 대기
            while (( current_jobs >= max_jobs_per_user )); do
                echo "Maximum job submission limit reached ($current_jobs/$max_jobs_per_user). Waiting for jobs to finish..."
                sleep 60  # 60초 대기 후 다시 확인
                current_jobs=$(squeue -u $USER | wc -l)
            done

            # 스케일러 사용 여부에 따른 실행 경로 및 인자 설정
            if [ "$use_scaler" == "true" ]; then
                # 스케일러 저장 경로 설정
                scaler_save_path="${model_save_path}/scalers/scaler_${fingerprint}_${model}.joblib"
                run_path="../../src/models/${model_subdir}"
                wrap_command="cd ${run_path} && python ${model}.py --fingerprint_type ${fingerprint} --file_path ${file_path} --model_save_path ${model_save_path} --scaler_save_path ${scaler_save_path}"
            else
                run_path="../../src/models/${model_subdir}"
                wrap_command="cd ${run_path} && python ${model}.py --fingerprint_type ${fingerprint} --file_path ${file_path} --model_save_path ${model_save_path}"
            fi

            # 작업 제출
            # 작업 제출
            sbatch --partition=gpu1,gpu4,cpu1 --cpus-per-task=15 \
                   --job-name=${target_name}_${fingerprint}_${model}_${EXPERIMENT_TYPE} \
                   --output=${log_dir}/${fingerprint}_${model}.log \
                   --error=${log_dir}/${fingerprint}_${model}.err \
                   --wrap="${wrap_command}"
        done
    done
done

#${EXPERIMENT_TYPE} - Experiment type: MF (Molecular Fingerprints), MD (Molecular Descriptors), or COMBINED (MF+MD)