import re
import os
import glob
import pandas as pd

# 1) 로그 파일이 있는 디렉토리 지정 (실제 경로에 맞게 수정)
log_dir = "/home2/jjy0605/Toxicity/0817_Genotoxicity/Paper/Model/Performance/Eco_model"

# 2) 로그 파일 패턴 (*.log)
log_files = glob.glob(os.path.join(log_dir, "*.log"))

# 3) 정규표현식 패턴 정의 (Test 지표와 Validation 지표)
test_patterns = {
    "F1": re.compile(r"Test F1 Score:\s*([0-9]*\.?[0-9]+)"),
    "Precision": re.compile(r"Test Precision:\s*([0-9]*\.?[0-9]+)"),
    "Recall": re.compile(r"Test Recall:\s*([0-9]*\.?[0-9]+)"),
    "Accuracy": re.compile(r"Test Accuracy:\s*([0-9]*\.?[0-9]+)"),
    "AUC": re.compile(r"Test AUC:\s*([0-9]*\.?[0-9]+)")
}
val_patterns = {
    "F1": re.compile(r"Validation F1 Score:\s*([0-9]*\.?[0-9]+)"),
    "Precision": re.compile(r"Validation Precision:\s*([0-9]*\.?[0-9]+)"),
    "Recall": re.compile(r"Validation Recall:\s*([0-9]*\.?[0-9]+)"),
    "Accuracy": re.compile(r"Validation Accuracy:\s*([0-9]*\.?[0-9]+)"),
    "AUC": re.compile(r"Validation AUC:\s*([0-9]*\.?[0-9]+)")
}

# 4) 결과 저장용 리스트
records = []

for file_path in log_files:
    file_name = os.path.basename(file_path)
    
    # 파일명에서 "TG", "FeatureSet", "Model" 추출
    # 예: "TG478_MACCS_logistic.log" -> TG = "TG478", FeatureSet = "MACCS", Model = "logistic"
    base_name, _ = os.path.splitext(file_name)
    parts = base_name.split("_")
    if len(parts) < 3:
        # 파일명이 기대한 형식이 아니면 건너뜁니다.
        continue
    tg = parts[0]
    feature_set = parts[1]
    model = parts[2]
    
    # 로그 파일 내용 읽기
    with open(file_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    
    # 각 Test 지표 추출
    test_metrics = {}
    for key, pattern in test_patterns.items():
        match = pattern.search(log_content)
        test_metrics[key] = float(match.group(1)) if match else None
        
    # 각 Validation 지표 추출
    val_metrics = {}
    for key, pattern in val_patterns.items():
        match = pattern.search(log_content)
        val_metrics[key] = float(match.group(1)) if match else None
    
    records.append({
        "TG": tg,
        "FeatureSet": feature_set,
        "Model": model,
        "Test_F1": test_metrics["F1"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_AUC": test_metrics["AUC"],
        "Val_F1": val_metrics["F1"],
        "Val_Precision": val_metrics["Precision"],
        "Val_Recall": val_metrics["Recall"],
        "Val_Accuracy": val_metrics["Accuracy"],
        "Val_AUC": val_metrics["AUC"]
    })

# 5) DataFrame 생성
df = pd.DataFrame(records)
print("Extracted Data:")
print(df)

# 6) 원하는 열 순서: "TG", "FeatureSet", "Model", 그 외 지표들
cols = ["TG", "FeatureSet", "Model", 
        "Test_F1", "Test_AUC", "Test_Precision", "Test_Recall", "Test_Accuracy", 
        "Val_F1", "Val_AUC",  "Val_Precision", "Val_Recall", "Val_Accuracy"]
df_final = df[cols].copy()

# 7) (정렬은 TG 순서를 지정할 필요 없으므로, FeatureSet과 Model에 대한 정렬이 필요할 경우만 적용)
# 만약 로그 파일명이 올바르게 추출되었다면, 아래 단계는 선택 사항입니다.
feature_order = ["MACCS", "Morgan", "RDKit", "Layered"]
model_order = ["dt", "rf", "gbt", "xgb", "logistic"]

df_final["FeatureSet"] = df_final["FeatureSet"].str.strip()
df_final["Model"] = df_final["Model"].str.strip().str.lower()

df_final["FeatureSet"] = pd.Categorical(df_final["FeatureSet"], categories=feature_order, ordered=True)
df_final["Model"] = pd.Categorical(df_final["Model"], categories=model_order, ordered=True)

df_final_sorted = df_final.sort_values(by=["FeatureSet", "Model"]).reset_index(drop=True)

# 8) 엑셀 파일로 저장
output_excel = "Eco_perf.xlsx"
df_final_sorted.to_excel(output_excel, index=False)
print("\nFinal Sorted Table:")
print(df_final_sorted)
print(f"\nSaved summary to {output_excel}")