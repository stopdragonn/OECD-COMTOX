import sys
sys.path.append('../../')

import json
import warnings
import joblib
import logging
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from utils.read_data import load_data
from utils.smiles2fing import Smiles2Fing
from utils.common import ParameterGrid
from rdkit import RDLogger
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# 표준 출력에 로그를 기록하기 위해 설정
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')

def save_results(result, path):
    with open(path, 'w') as f:
        json.dump(result, f)

def find_best_model(results, metric='f1', metric_agg='mean'):
    best_model = None
    best_score = -np.inf
    best_model_key = None
    
    for model_key in results['model'].keys():
        scores = results[metric][model_key]
        if metric_agg == 'mean':
            agg_score = np.mean(scores)
        elif metric_agg == 'median':
            agg_score = np.median(scores)
        else:
            raise ValueError("metric_agg must be either 'mean' or 'median'")
        
        if agg_score > best_score:
            best_score = agg_score
            best_model = results['model'][model_key]
            best_model_key = model_key
    
    return best_model_key, best_model, best_score


def main(fingerprint_type, file_path, model_save_path, scaler_save_path):
    # 데이터를 로드하고 train/test로 4:1로 분할
    x, y, scaler = load_data(fingerprint_type=fingerprint_type, file_path=file_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

    params_dict = {
        'reg_param': np.append(np.array([0]), np.logspace(-5, 0, 10)),
        'tol': np.logspace(-5, -3, 10)
    }
    params = ParameterGrid(params_dict)

    result = {'model': {}, 'precision': {}, 'recall': {}, 'f1': {}, 'accuracy': {}, 'roc_auc': {}}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 훈련 데이터셋에서 5-fold 교차검증 수행
    for p in tqdm(range(len(params))):
        model_key = f'model{p}'
        result['model'][model_key] = params[p]
        result['precision'][model_key] = []
        result['recall'][model_key] = []
        result['f1'][model_key] = []
        result['accuracy'][model_key] = []
        result['roc_auc'][model_key] = []

        for train_idx, val_idx in kf.split(x_train, y_train):
            fold_train_x, fold_val_x = x_train.iloc[train_idx], x_train.iloc[val_idx]
            fold_train_y, fold_val_y = y_train.iloc[train_idx], y_train.iloc[val_idx]

            sm = SMOTE(random_state=42)
            fold_train_x, fold_train_y = sm.fit_resample(fold_train_x, fold_train_y)

            model = QuadraticDiscriminantAnalysis(**params[p])
            model.fit(fold_train_x, fold_train_y)
            pred_probs = model.predict_proba(fold_val_x)[:, 1]  # 확률 예측값 사용
            pred = model.predict(fold_val_x)
            
            result['precision'][model_key].append(precision_score(fold_val_y, pred))
            result['recall'][model_key].append(recall_score(fold_val_y, pred))
            result['f1'][model_key].append(f1_score(fold_val_y, pred))
            result['accuracy'][model_key].append(accuracy_score(fold_val_y, pred))
            result['roc_auc'][model_key].append(roc_auc_score(fold_val_y, pred_probs))  # AUC 계산 시 확률 사용

        # 중간 결과 저장
        save_results(result, f'{model_save_path}/qda_intermediate_{fingerprint_type}.json')

    # 최적 모델 찾기
    best_model_key, best_model, best_f1_score = find_best_model(result, metric='f1')

    best_precision = np.mean(result['precision'][best_model_key])
    best_recall = np.mean(result['recall'][best_model_key])
    best_accuracy = np.mean(result['accuracy'][best_model_key])
    best_roc_auc = np.mean(result['roc_auc'][best_model_key])

    logging.info(f"Best Model Parameters: {best_model}")
    logging.info(f"Validation F1 Score: {best_f1_score}")
    logging.info(f"Validation Precision: {best_precision}")
    logging.info(f"Validation Recall: {best_recall}")
    logging.info(f"Validation Accuracy: {best_accuracy}")
    logging.info(f"Validation AUC: {best_roc_auc}")

    # 최적 모델을 전체 훈련 데이터셋에 대해 다시 학습하고 테스트 셋에서 평가
    final_model = QuadraticDiscriminantAnalysis(**best_model)
    final_model.fit(x_train, y_train)
    final_pred_probs = final_model.predict_proba(x_test)[:, 1]  # 확률 예측값 사용
    final_pred = final_model.predict(x_test)

    test_precision = precision_score(y_test, final_pred)
    test_recall = recall_score(y_test, final_pred)
    test_f1 = f1_score(y_test, final_pred)
    test_accuracy = accuracy_score(y_test, final_pred)
    test_roc_auc = roc_auc_score(y_test, final_pred_probs)  # AUC 계산 시 확률 사용

    logging.info(f"Test F1 Score: {test_f1}")
    logging.info(f"Test Precision: {test_precision}")
    logging.info(f"Test Recall: {test_recall}")
    logging.info(f"Test Accuracy: {test_accuracy}")
    logging.info(f"Test AUC: {test_roc_auc}")

    # 최적 모델 저장
    model_filename = f'{model_save_path}/best_model_{fingerprint_type}_qda.joblib'
    joblib.dump(final_model, model_filename)
    logging.info(f"Best model saved as {model_filename} with F1 score: {test_f1}")

    # 최적 스케일러 저장
    joblib.dump(scaler, scaler_save_path)
    logging.info(f"Scaler saved as {scaler_save_path}")

if __name__ == '__main__':
    # 메인 함수에 fingerprint_type, 파일 경로, 모델 저장 경로, 스케일러 저장 경로를 전달
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingerprint_type', type=str, default='MACCS', help='Type of molecular fingerprint to use')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the Excel file with data')
    parser.add_argument('--model_save_path', type=str, required=True, help='Directory where the model will be saved')
    parser.add_argument('--scaler_save_path', type=str, help='Path to save the scaler')
    args = parser.parse_args()
    main(args.fingerprint_type, args.file_path, args.model_save_path, args.scaler_save_path)