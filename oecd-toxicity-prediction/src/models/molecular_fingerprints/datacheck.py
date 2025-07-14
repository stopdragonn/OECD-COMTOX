import openpyxl
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
try: 
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
    
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from utils.read_data import load_data


def prepare_data(file_path, fingerprint_type='MACCS'):
    """
    1. 엑셀 파일을 읽어 fingerprint와 라벨 데이터를 로드한다.
    2. 전체 데이터를 Train+Validation (80%)와 Test (20%)로 분할한다.
    3. Train+Validation 내에서 Stratified 5-Fold CV를 수행하고,
       각 Fold의 Train set에 SMOTE를 적용하여 oversampling된 데이터를 생성한다.
       
    최종적인 데이터 분할 비율은 Train:Validation:Test = 0.64:0.16:0.2 임.
    
    Returns:
        dict: 결과 딕셔너리로, 각 Fold의 oversampled train set과 함께
              오버샘플링 적용 전 원본 Train set의 라벨별 분포도 포함.
    """
    # 1. 데이터 로드
    X, y = load_data(file_path=file_path, fingerprint_type=fingerprint_type)
    
    # 2. 전체 데이터를 Train+Validation (80%)와 Test (20%)로 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    # Stratified 5-Fold CV 수행
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = []
    
    for train_idx, val_idx in skf.split(X_train_val, y_train_val):
        X_train_fold = X_train_val.iloc[train_idx]
        y_train_fold = y_train_val.iloc[train_idx]
        X_val_fold = X_train_val.iloc[val_idx]
        y_val_fold = y_train_val.iloc[val_idx]
        
        # 원본 Train set의 라벨 분포 (오버샘플링 적용 전)
        original_label_distribution = pd.Series(y_train_fold).value_counts().to_dict()
        
        # SMOTE 적용: 소수 클래스의 샘플을 증대하여 클래스 균형 맞추기
        sm = SMOTE(random_state=42)
        X_train_fold_over, y_train_fold_over = sm.fit_resample(X_train_fold, y_train_fold)
        
        cv_splits.append({
            'X_train': X_train_fold,
            'y_train': y_train_fold,
            'X_val': X_val_fold,
            'y_val': y_val_fold,
            'X_train_over': X_train_fold_over,
            'y_train_over': y_train_fold_over,
            'oversampled_train_count': len(y_train_fold_over),
            'original_label_distribution': original_label_distribution
        })
    
    result = {
        'test_set': {'X_test': X_test, 'y_test': y_test},
        'cv_splits': cv_splits,
        'train_val_count': len(y_train_val),
        'test_count': len(y_test)
    }
    
    return result

if __name__ == '__main__':
    file_path = '../../data/raw/molecular_fingerprints/TG201.xlsx'
    fingerprint_type = 'MACCS'
    
    data_splits = prepare_data(file_path, fingerprint_type=fingerprint_type)
    
    print("Train+Validation set count:", data_splits['train_val_count'])
    print("Test set count:", data_splits['test_count'])
    for i, fold in enumerate(data_splits['cv_splits']):
        print(f"Fold {i+1} - Oversampled Train set count: {fold['oversampled_train_count']}")
        print(f"Fold {i+1} - Original Label Distribution (pre-SMOTE): {fold['original_label_distribution']}")
