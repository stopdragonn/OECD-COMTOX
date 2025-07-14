#!/bin/bash

# 모델 리스트 정의
models=('gbt' 'logistic' 'xgb') # 'mlp' 'dt' 'lgb' 'rf' 'lda' 'plsda' 'qda' 
fingerprints=("MACCS" "Morgan" "RDKit" "Layered") # "Pattern"  
file_path='../../data/TG471.xlsx'
model_save_path='../../results/1002/TG471'

# 로그 디렉토리 생성
mkdir -p logs/1002/TG471
mkdir -p tg471/results/1002/TG471
# 각 모델에 대해 실험 실행
for model in "${models[@]}"; do
    for fingerprint in "${fingerprints[@]}"; do
        echo "Submitting job for model: $model with fingerprint: $fingerprint"

        sbatch --partition=gpu1 --gres=gpu:1 --cpus-per-task=15 \
               --job-name=TG471_${fingerprint}_${model} \
               --output=logs/1002/TG471/${fingerprint}_${model}.log \
               --error=logs/1002/TG471/${fingerprint}_${model}.err \
               --wrap="cd tg471/run && python ${model}.py --fingerprint_type ${fingerprint} --file_path ${file_path} --model_save_path ${model_save_path}"  
    done
done

#MF (SCALER X)

