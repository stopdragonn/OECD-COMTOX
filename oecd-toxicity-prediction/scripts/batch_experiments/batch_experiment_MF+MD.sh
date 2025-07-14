#!/bin/bash

# 모델 리스트 정의
models=('rf' 'gbt' 'l            # 스케일러 저장 경로 설정
            scaler_save_path="${model_save_path}/scalers/scaler_${fingerprint}_${model}.joblib"

            sbatch --partition=gpu1 --gres=gpu:1 --cpus-per-task=15 \
                --job-name=${target_name}_${fingerprint}_${model} \
                --output=${log_dir}/${fingerprint}_${model}.log \
                --error=${log_dir}/${fingerprint}_${model}.err \
                --wrap="cd ../../src/models/combined && python ${model}.py --fingerprint_type ${fingerprint} --file_path ${file_path} --model_save_path ${model_save_path} --scaler_save_path ${scaler_save_path}" 'xgb' 'dt') # 'mlp' 'lgb' 'lda' 'plsda' 'qda'
fingerprints=("MACCS" "Morgan" "RDKit" "Layered") # "Pattern"

# 여러 타겟 파일 리스트 정의
file_paths=('../../data/raw/combined/TG414_Descriptor_desalt_250501.xlsx' '../../data/raw/combined/TG416_Descriptor_desalt_250501.xlsx' '../../data/raw/combined/TG453_Descriptor_desalt_250501.xlsx')

# 모델 저장 경로
base_model_save_path='../../results/models/combined'
base_log_path='../../results/logs/combined'

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
            echo "Submitting job for target: $file_path, model: $model, fingerprint: $fingerprint"

            # 제출된 작업 수 확인
            current_jobs=$(squeue -u $USER | wc -l)

            # 제출된 작업 수가 최대값에 도달하면 대기
            while (( current_jobs >= max_jobs_per_user )); do
                echo "Maximum job submission limit reached ($current_jobs/$max_jobs_per_user). Waiting for jobs to finish..."
                sleep 60  # 60초 대기 후 다시 확인
                current_jobs=$(squeue -u $USER | wc -l)
            done

            # 스케일러 저장 경로 설정
            scaler_save_path="${model_save_path}/scaler_${fingerprint}_${model}.joblib"

            sbatch --partition=gpu1 --gres=gpu:1 --cpus-per-task=15 \
                --job-name=${target_name}_${fingerprint}_${model} \
                --output=${log_dir}/${fingerprint}_${model}.log \
                --error=${log_dir}/${fingerprint}_${model}.err \
                --wrap="cd tg471/run/scaler && python ${model}.py --fingerprint_type ${fingerprint} --file_path ${file_path} --model_save_path ${model_save_path} --scaler_save_path ${scaler_save_path}"  
        done
    done
done
#MF+MD (SCALER O)