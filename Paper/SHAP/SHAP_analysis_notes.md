# SHAP 분석 결과 설명

## 모델 설명
원본 모델(`TG403_aerosol_best_model_RDKit_rf.joblib`)은 scikit-learn 1.5.1 버전에서 훈련되었으나, 
현재 환경은 scikit-learn 0.24.2 버전을 사용하고 있어 버전 호환성 문제로 불러올 수 없었습니다.

따라서 다음과 같은 대체 접근 방식을 사용했습니다:
1. 동일한 데이터셋(`TG403aerosolINPUT_desalt_wSMILES.xlsx`)에서 RDKit 분자 지문 생성
2. 이 데이터를 사용하여 새 RandomForest 모델 생성 (`n_estimators=100, random_state=42`)
3. 생성된 모델에 SHAP 분석 적용

## 결과 해석 시 주의사항
- 대체 모델은 원본 모델과 동일한 훈련/검증 데이터 분할을 사용하지 않았으므로, 
  특성 중요도 순위가 원본 모델과 다를 수 있습니다.
- 현재 모델은 동일한 데이터셋 내에서만 검증되었으므로, 
  외부 검증 없이 특성 중요도를 해석할 때 주의가 필요합니다.
- 논문에서 이 결과를 인용할 때는 대체 모델을 사용했다는 점을 반드시 명시해야 합니다.

## SHAP 분석 결과
SHAP 분석은 모델의 예측에 각 특성이 어떻게 기여하는지 보여줍니다. 
상위 10개 특성(RDKit 지문)의 중요도는 다음과 같습니다:

1. rdkit_193: 0.004484
2. rdkit_222: 0.004034
3. rdkit_1383: 0.003791
4. rdkit_1258: 0.003324
5. rdkit_1597: 0.002930
6. rdkit_1447: 0.002822
7. rdkit_765: 0.002763
8. rdkit_1115: 0.002704
9. rdkit_888: 0.002630
10. rdkit_1436: 0.002489

## 바 차트와 비즈웜 플롯
- `TG403aerosolINPUT_desalt_wSMILES_RDKit_shap_top10_bar.png`: 상위 10개 특성의 중요도를 보여주는 바 차트
- `TG403aerosolINPUT_desalt_wSMILES_RDKit_shap_beeswarm.png`: 각 특성이 예측에 미치는 영향과 분포를 보여주는 비즈웜 플롯
- `TG403aerosolINPUT_desalt_wSMILES_RDKit_feature_importance.csv`: 모든 특성의 중요도를 담은 CSV 파일
