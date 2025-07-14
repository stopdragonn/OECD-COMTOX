import sys
sys.path.append('../')
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from utils.smiles2fing import Smiles2Fing
from utils.read_data import load_data

def main(fingerprint_type, file_path, model_filename): #, scaler_save_path
    # 스케일러 로드
    #scaler = joblib.load(scaler_save_path)

    # 데이터를 로드하고 train/test로 4:1로 분할
    x, y, _ = load_data(fingerprint_type=fingerprint_type, file_path=file_path) #, scaler=scaler
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

    # 모델 로드
    final_model = joblib.load(model_filename)

    # 테스트 셋에 대한 예측
    final_pred = final_model.predict(x_test)

    # 성능 지표 계산
    test_precision = precision_score(y_test, final_pred)
    test_recall = recall_score(y_test, final_pred)
    test_f1 = f1_score(y_test, final_pred)
    test_accuracy = accuracy_score(y_test, final_pred)

    # AUC 계산
    test_probs = final_model.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    # 결과 출력
    print(f"Test F1 Score: {test_f1}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test AUC: {test_auc}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingerprint_type', type=str, default='MACCS', help='Type of molecular fingerprint to use')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the Excel file with data')
    parser.add_argument('--model_filename', type=str, required=True, help='Directory where the model will be saved')
    #parser.add_argument('--scaler_save_path', type=str, required=True, help='Path to save the scaler')
    args = parser.parse_args()
    main(args.fingerprint_type, args.file_path, args.model_filename) #, args.scaler_save_path

# 스케일러 쓸 때
# def load_data(file_path, fingerprint_type='MACCS', scaler=None, scaler_save_path=None):
#     """
#     엑셀 파일에서 데이터를 로드하고 SMILES와 Molecular Descriptor를 처리하는 함수.
    
#     Args:
#         file_path (str): 엑셀 파일의 경로
#         fingerprint_type (str): 사용할 fingerprint의 유형
#         scaler (StandardScaler, optional): 학습된 스케일러. 지정되지 않으면 새로운 스케일러를 학습하고 저장함.
#         scaler_save_path (str, optional): 새로운 스케일러를 저장할 경로. 지정되지 않으면 저장하지 않음.
        
#     Returns:
#         pd.DataFrame, pd.Series, StandardScaler: 전처리된 특징 데이터와 라벨, 스케일러 객체
#     """
#     df = pd.read_excel(file_path)

#     # Molecular Fingerprints 생성
#     drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)

#     # Molecular Descriptor와 Fingerprints 결합
#     descriptors = df.drop(columns=['Chemical', 'CasRN', 'SMILES', 'Genotoxicity', 'SMILES']).drop(index=drop_idx).reset_index(drop=True)
#     features = pd.concat([descriptors, fingerprints], axis=1)

#     if scaler is None:
#         # 학습 시 새로운 스케일러를 학습하고 저장
#         scaler = StandardScaler()
#         scaled_features = scaler.fit_transform(features)
#         if scaler_save_path:
#             joblib.dump(scaler, scaler_save_path)
#     else:
#         # 테스트 시 저장된 스케일러를 불러와서 사용
#         scaled_features = scaler.transform(features)

#     y = df.Genotoxicity.drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop=True)

#     return pd.DataFrame(scaled_features, columns=features.columns), y, scaler


#스케일러 쓸 때
# def main(fingerprint_type, file_path, model_filename, scaler_save_path): #
#     # 스케일러 로드
#     scaler = joblib.load(scaler_save_path)

#     # 데이터를 로드하고 train/test로 4:1로 분할
#     x, y, _ = load_data(fingerprint_type=fingerprint_type, file_path=file_path) #, scaler=scaler
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

#     # 모델 로드
#     final_model = joblib.load(model_filename)

#     # 테스트 셋에 대한 예측
#     final_pred = final_model.predict(x_test)

#     # 성능 지표 계산
#     test_precision = precision_score(y_test, final_pred)
#     test_recall = recall_score(y_test, final_pred)
#     test_f1 = f1_score(y_test, final_pred)
#     test_accuracy = accuracy_score(y_test, final_pred)

#     # AUC 계산
#     test_probs = final_model.predict_proba(x_test)[:, 1]
#     test_auc = roc_auc_score(y_test, test_probs)

#     # 결과 출력
#     print(f"Test F1 Score: {test_f1}")
#     print(f"Test Precision: {test_precision}")
#     print(f"Test Recall: {test_recall}")
#     print(f"Test Accuracy: {test_accuracy}")
#     print(f"Test AUC: {test_auc}")

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--fingerprint_type', type=str, default='MACCS', help='Type of molecular fingerprint to use')
#     parser.add_argument('--file_path', type=str, required=True, help='Path to the Excel file with data')
#     parser.add_argument('--model_filename', type=str, required=True, help='Directory where the model will be saved')
#     parser.add_argument('--scaler_save_path', type=str, required=True, help='Path to save the scaler')
#     args = parser.parse_args()
#     main(args.fingerprint_type, args.file_path, args.model_filename, args.scaler_save_path) #
