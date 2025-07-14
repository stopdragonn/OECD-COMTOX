# [No Scaler_MF]

import openpyxl
import pandas as pd
from .smiles2fing import Smiles2Fing

def load_data(file_path, fingerprint_type='MACCS'):
    """
    엑셀 파일에서 데이터를 로드하고 SMILES 문자열을 fingerprint로 변환하는 함수
    
    Args:
        file_path (str): 엑셀 파일의 경로
        fingerprint_type (str): 사용할 fingerprint의 유형
        
    Returns:
        list, pd.Series: 변환된 fingerprint와 대응되는 라벨 데이터
    """
    df = pd.read_excel(file_path)
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)
    
    # 데이터 타입 확인
    if df.Toxicity.dtype == 'object':  # 문자열 타입인 경우
        y = df.Toxicity.drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop=True)
    else:  # 이미 숫자 타입인 경우
        y = df.Toxicity.drop(drop_idx).reset_index(drop=True)
          
    return fingerprints, y


# [Scaler_MF+MD]

# import openpyxl
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from .smiles2fing import Smiles2Fing

# def save_scaler(scaler, file_path):
#     """
#     스케일러 객체를 파일로 저장하는 함수
    
#     Args:
#         scaler (StandardScaler): 훈련 데이터에 맞게 적합된 스케일러 객체
#         file_path (str): 스케일러를 저장할 파일 경로
#     """
#     joblib.dump(scaler, file_path)
#     print(f"Scaler saved to {file_path}")

# # scaler 적용할 feature에 대해서
# def load_data(file_path, fingerprint_type='MACCS', scaler_save_path=None):
#     """
#     엑셀 파일에서 데이터를 로드하고 SMILES와 Molecular Descriptor를 처리하는 함수
    
#     Args:
#         file_path (str): 엑셀 파일의 경로
#         fingerprint_type (str): 사용할 fingerprint의 유형
#         scaler_save_path (str, optional): 스케일러를 저장할 경로. 지정되지 않으면 스케일러를 저장하지 않음.
        
#     Returns:
#         pd.DataFrame, pd.Series, StandardScaler: 전처리된 특징 데이터, 라벨, 그리고 적합된 스케일러 객체
#     """
#     df = pd.read_excel(file_path)

#     # Molecular Fingerprints 생성
#     drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)

#     # Molecular Descriptor와 Fingerprints 결합
#     descriptors = df.drop(columns=['SMILES', 'DTXSID', 'Toxicity']).drop(index=drop_idx).reset_index(drop=True)
#     features = pd.concat([descriptors, fingerprints], axis=1)

#     # 데이터프레임의 head 출력
#     #print("Combined Features DataFrame head:\n", features.head())

#     # 스케일링
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)

#     # 스케일러 저장 (필요한 경우)
#     if scaler_save_path:
#         save_scaler(scaler, scaler_save_path)

#     y = df.Genotoxicity.drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop=True)

#     return pd.DataFrame(scaled_features, columns=features.columns), y, scaler



# [Scaler_MD]

# import openpyxl
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from .smiles2fing import Smiles2Fing

# def save_scaler(scaler, file_path):
#     """
#     스케일러 객체를 파일로 저장하는 함수
    
#     Args:
#         scaler (StandardScaler): 훈련 데이터에 맞게 적합된 스케일러 객체
#         file_path (str): 스케일러를 저장할 파일 경로
#     """
#     joblib.dump(scaler, file_path)
#     print(f"Scaler saved to {file_path}")

# # scaler 적용할 feature에 대해서
# def load_data(file_path, fingerprint_type='MACCS', scaler_save_path=None):
#     """
#     엑셀 파일에서 데이터를 로드하고 Molecular Descriptor를 처리하는 함수
    
#     Args:
#         file_path (str): 엑셀 파일의 경로
#         fingerprint_type (str): 사용할 fingerprint의 유형
#         scaler_save_path (str, optional): 스케일러를 저장할 경로. 지정되지 않으면 스케일러를 저장하지 않음.
        
#     Returns:
#         pd.DataFrame, pd.Series, StandardScaler: 전처리된 특징 데이터, 라벨, 그리고 적합된 스케일러 객체
#     """
#     df = pd.read_excel(file_path)

#     # # Molecular Fingerprints 생성
#     drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)

#     # Molecular Descriptor와 Fingerprints 결합
#     descriptors = df.drop(columns=['Chemical', 'CasRN', 'Genotoxicity', 'SMILES']).drop(index=drop_idx).reset_index(drop=True)
#     #features에 fingerprints는 concat안함.
#     features = descriptors

#     # 데이터프레임의 head 출력
#     #print("Combined Features DataFrame head:\n", features.head())

#     # 스케일링
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)

#     # 스케일러 저장 (필요한 경우)
#     if scaler_save_path:
#         save_scaler(scaler, scaler_save_path)

#     y = df.Genotoxicity.drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop=True)

#     return pd.DataFrame(scaled_features, columns=features.columns), y, scaler


# [No Scaler_ToxPrint(인풋에 이미 분자지문이 포함되어있음)]

# import openpyxl
# import pandas as pd

# def load_data(file_path):
#     """
#     엑셀 파일에서 데이터를 로드하고 ToxPrint 분자지문을 반환하는 함수
    
#     Args:
#         file_path (str): 엑셀 파일의 경로
        
#     Returns:
#         pd.DataFrame, pd.Series: ToxPrint 분자지문과 대응되는 라벨 데이터
#     """
#     df = pd.read_excel(file_path, sheet_name='consv')
    
#     # ToxPrint 분자지문을 포함한 열을 범위로 선택 (5번째 열부터 마지막에서 두 번째 열까지)
#     fingerprints = df.iloc[:, 4:-1]  # 5번째 열부터 마지막에서 두 번째 열까지 선택
    
#     # 라벨 데이터 생성
#     y = df.consensus_consv.replace({'negative': 0, 'positive': 1})
    
#     return fingerprints, y