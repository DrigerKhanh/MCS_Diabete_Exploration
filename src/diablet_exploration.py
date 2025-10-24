import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("../dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

print("=== PHÂN TÍCH IMBALANCE ===")
imbalance_analysis = df['Diabetes_binary'].value_counts(normalize=True)
print(imbalance_analysis)

# ========== THÔNG MINH HƠN: CHỈ CÂN BẰNG KHI CẦN THIẾT ==========

if imbalance_analysis[1] < 0.3:  # Nếu tỷ lệ tiểu đường < 30% (imbalance)
    print("\n⚠️ Dataset bị imbalance - thực hiện cân bằng...")

    # Lấy tất cả bệnh nhân tiểu đường + sampling ngẫu nhiên từ không bệnh
    diabetes_cases = df[df['Diabetes_binary'] == 1]
    non_diabetes_cases = df[df['Diabetes_binary'] == 0]

    # Kiểm tra xem có đủ samples không
    available_non_diabetes = len(non_diabetes_cases)
    needed_samples = min(len(diabetes_cases) * 2, available_non_diabetes)

    non_diabetes_sample = non_diabetes_cases.sample(n=needed_samples, random_state=42)

    # Tạo dataset cân bằng
    balanced_df = pd.concat([diabetes_cases, non_diabetes_sample], ignore_index=True)
    print(f"Dataset sau khi cân bằng: {balanced_df.shape}")

else:
    print("\n✅ Dataset đã cân bằng - sử dụng trực tiếp")
    balanced_df = df.copy()

print("Phân bố sau xử lý:")
print(balanced_df['Diabetes_binary'].value_counts(normalize=True))

# ========== PHẦN CÒN LẠI GIỮ NGUYÊN ==========
association_df = balanced_df.copy()

# Chuyển các biến quan trọng thành binary categories
association_df['BMI_High'] = (association_df['BMI'] >= 25).astype(int)
association_df['BMI_VeryHigh'] = (association_df['BMI'] >= 30).astype(int)
association_df['Age_45plus'] = (association_df['Age'] >= 9).astype(int)
association_df['Age_65plus'] = (association_df['Age'] >= 13).astype(int)
association_df['Health_Poor'] = (association_df['GenHlth'] >= 4).astype(int)
association_df['PhysHealth_Poor'] = (association_df['PhysHlth'] > 7).astype(int)

# Tạo binary dataset
binary_columns = [
    'HighBP', 'HighChol', 'Smoker', 'HeartDiseaseorAttack',
    'PhysActivity', 'HvyAlcoholConsump', 'DiffWalk',
    'BMI_High', 'BMI_VeryHigh', 'Age_45plus', 'Age_65plus',
    'Health_Poor', 'PhysHealth_Poor'
]

binary_data = association_df[binary_columns].copy()
binary_data['Has_Diabetes'] = association_df['Diabetes_binary']
binary_data = binary_data.astype(bool)

print("\n=== BINARY DATASET ===")
print(f"Kích thước: {binary_data.shape}")
print("Phân bố Diabetes:", binary_data['Has_Diabetes'].value_counts())

# ========== APRIORI ==========
print("\n=== TÌM FREQUENT ITEMSETS ===")
frequent_itemsets = apriori(binary_data, min_support=0.01, use_colnames=True, max_len=4)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(f"Tìm thấy {len(frequent_itemsets)} frequent itemsets")

# ========== ASSOCIATION RULES ==========
print("\n=== TÌM ASSOCIATION RULES ===")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
print(f"Tìm thấy {len(rules)} rules")

# Lọc rules về tiểu đường
diabetes_rules = rules[rules['consequents'].apply(lambda x: 'Has_Diabetes' in x)]

print(f"\n=== TÌM THẤY {len(diabetes_rules)} RULES VỀ TIỂU ĐƯỜNG ===")

if len(diabetes_rules) > 0:
    diabetes_rules_sorted = diabetes_rules.sort_values('confidence', ascending=False)

    print("TOP 10 RULES VỀ TIỂU ĐƯỜNG:")
    for i, rule in diabetes_rules_sorted.head(10).iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        print(f"Rule {i + 1}: NẾU {antecedents} → THÌ {consequents}")
        print(f"   Confidence: {rule['confidence']:.3f} | Support: {rule['support']:.3f} | Lift: {rule['lift']:.3f}")
        print()

print("✅ HOÀN THÀNH ASSOCIATION RULE MINING")