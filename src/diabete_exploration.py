import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("../dataset/diabetes.csv")

print("=== PHÂN TÍCH IMBALANCE ===")
imbalance_analysis = df['Outcome'].value_counts(normalize=True)
print(imbalance_analysis)

# ========== XỬ LÝ IMBALANCE ==========
if imbalance_analysis[1] < 0.3:  # Nếu tỷ lệ tiểu đường < 30%
    print("\n⚠️ Dataset bị imbalance - thực hiện cân bằng...")

    diabetes_cases = df[df['Outcome'] == 1]
    non_diabetes_cases = df[df['Outcome'] == 0]

    # Lấy mẫu cân bằng
    non_diabetes_sample = non_diabetes_cases.sample(
        n=len(diabetes_cases),
        random_state=42
    )

    balanced_df = pd.concat([diabetes_cases, non_diabetes_sample], ignore_index=True)
    print(f"Dataset sau khi cân bằng: {balanced_df.shape}")
else:
    print("\n✅ Dataset đã cân bằng - sử dụng trực tiếp")
    balanced_df = df.copy()

print("Phân bố sau xử lý:")
print(balanced_df['Outcome'].value_counts(normalize=True))

# ========== CHUẨN BỊ DỮ LIỆU CHO ASSOCIATION RULES ==========
association_df = balanced_df.copy()

# Tạo các biến binary từ clinical measures
association_df['High_Glucose'] = (association_df['Glucose'] >= 140).astype(int)  # Tiêu chuẩn y tế
association_df['High_BP'] = (association_df['BloodPressure'] >= 90).astype(int)  # Huyết áp cao
association_df['Obese'] = (association_df['BMI'] >= 30).astype(int)  # Béo phì
association_df['Overweight'] = (association_df['BMI'] >= 25).astype(int)  # Thừa cân
association_df['High_Insulin'] = (association_df['Insulin'] >= 150).astype(int)  # Insulin cao
association_df['Age_40plus'] = (association_df['Age'] >= 40).astype(int)  # Tuổi > 40
association_df['High_Risk_Pedigree'] = (association_df['DiabetesPedigreeFunction'] >= 0.8).astype(int)
association_df['Multiple_Pregnancies'] = (association_df['Pregnancies'] >= 3).astype(int)

# Tạo binary dataset
binary_columns = [
    'High_Glucose', 'High_BP', 'Obese', 'Overweight',
    'High_Insulin', 'Age_40plus', 'High_Risk_Pedigree',
    'Multiple_Pregnancies'
]

binary_data = association_df[binary_columns].copy()
binary_data['Has_Diabetes'] = association_df['Outcome']
binary_data = binary_data.astype(bool)

print("\n=== BINARY DATASET CHO ASSOCIATION RULES ===")
print(f"Kích thước: {binary_data.shape}")
print("Mô tả các biến binary:")
for col in binary_columns:
    true_count = binary_data[col].sum()
    print(f"- {col}: {true_count} samples ({true_count / len(binary_data) * 100:.1f}%)")

print(f"Phân bố Diabetes: {binary_data['Has_Diabetes'].value_counts()}")

# ========== APRIORI ALGORITHM ==========
print("\n=== TÌM FREQUENT ITEMSETS ===")
frequent_itemsets = apriori(binary_data, min_support=0.05, use_colnames=True, max_len=3)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(f"Tìm thấy {len(frequent_itemsets)} frequent itemsets")
print("\nTop 10 frequent itemsets:")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

# ========== ASSOCIATION RULES ==========
print("\n=== TÌM ASSOCIATION RULES ===")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(f"Tìm thấy {len(rules)} rules")

# Lọc rules về tiểu đường
diabetes_rules = rules[rules['consequents'].apply(lambda x: 'Has_Diabetes' in x)]

print(f"\n=== TÌM THẤY {len(diabetes_rules)} RULES VỀ TIỂU ĐƯỜNG ===")

if len(diabetes_rules) > 0:
    diabetes_rules_sorted = diabetes_rules.sort_values('confidence', ascending=False)

    print("TOP 10 RULES VỀ TIỂU ĐƯỜNG:")
    print("=" * 70)

    for i, rule in diabetes_rules_sorted.head(10).iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])

        print(f"Rule {i + 1}:")
        print(f"   NẾU {antecedents}")
        print(f"   → THÌ {consequents}")
        print(f"   Confidence: {rule['confidence']:.3f} | Support: {rule['support']:.3f} | Lift: {rule['lift']:.3f}")
        print("-" * 50)

# ========== PHÂN TÍCH CÁC YẾU TỐ NGUY CƠ ==========
print("\n=== PHÂN TÍCH YẾU TỐ NGUY CƠ ===")
risk_factors = binary_columns.copy()
risk_analysis = []

for factor in risk_factors:
    diabetes_with_factor = binary_data[binary_data[factor]]['Has_Diabetes'].mean()
    diabetes_without_factor = binary_data[~binary_data[factor]]['Has_Diabetes'].mean()
    risk_ratio = diabetes_with_factor / diabetes_without_factor if diabetes_without_factor > 0 else float('inf')

    risk_analysis.append({
        'Risk_Factor': factor,
        'Diabetes_Rate_With_Factor': diabetes_with_factor,
        'Diabetes_Rate_Without_Factor': diabetes_without_factor,
        'Risk_Ratio': risk_ratio
    })

risk_df = pd.DataFrame(risk_analysis).sort_values('Risk_Ratio', ascending=False)

print("CÁC YẾU TỐ NGUY CƠ THEO TỶ LỆ RỦI RO:")
for _, row in risk_df.iterrows():
    print(f"- {row['Risk_Factor']}: Risk Ratio = {row['Risk_Ratio']:.2f}x")

# Visualize risk factors
plt.figure(figsize=(12, 6))
sns.barplot(x='Risk_Ratio', y='Risk_Factor', data=risk_df, palette='Reds_r')
plt.title('TỶ LỆ RỦI RO CỦA CÁC YẾU TỐ NGUY CƠ')
plt.xlabel('Tỷ lệ rủi ro (Risk Ratio)')
plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n✅ HOÀN THÀNH KHAI PHÁ YẾU TỐ NGUY CƠ TIỂU ĐƯỜNG")