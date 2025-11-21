import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu Pima Indians
df = pd.read_csv("../dataset/diabetes.csv")

# 1️⃣ Tổng quan dữ liệu
print("=== TỔNG QUAN DỮ LIỆU PIMA INDIANS ===")
print(f"Kích thước dataset: {df.shape}")
print(df.info())
print("\n5 dòng đầu:")
print(df.head())

# 2️⃣ Phân bố nhãn bệnh
print("\n=== PHÂN BỐ NHÃN BỆNH ===")
print("0 = Không tiểu đường, 1 = Có tiểu đường")
outcome_dist = df['Outcome'].value_counts(normalize=True)
print(outcome_dist)

# 3️⃣ Thống kê mô tả
print("\n=== THỐNG KÊ MÔ TẢ ===")
print(df.describe())

# 4️⃣ Kiểm tra missing values (được mã hóa thành 0)
print("\n=== KIỂM TRA GIÁ TRỊ 0 (CÓ THỂ LÀ MISSING) ===")
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count} giá trị 0 ({zero_count/len(df)*100:.1f}%)")

# 5️⃣ So sánh trung bình giữa 2 nhóm
print("\n=== SO SÁNH TRUNG BÌNH THEO NHÓM BỆNH ===")
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
mean_comparison = df.groupby('Outcome')[features].mean()
print(mean_comparison)

# 6️⃣ Phân tích tương quan
print("\n=== PHÂN TÍCH TƯƠNG QUAN ===")
corr = df.corr(numeric_only=True)
corr_target = corr['Outcome'].sort_values(ascending=False)

print("🔍 TƯƠNG QUAN VỚI OUTCOME:")
print(corr_target)

# ========================= TRỰC QUAN HÓA =========================

# Biểu đồ phân bố Outcome
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=df)
plt.title('PHÂN BỐ BỆNH TIỂU ĐƯỜNG TRONG DATASET')
plt.xlabel('Kết quả (0: Không bệnh, 1: Có bệnh)')
plt.ylabel('Số lượng')
plt.show()

# Heatmap tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('HEATMAP TƯƠNG QUAN GIỮA CÁC BIẾN LÂM SÀNG')
plt.tight_layout()
plt.show()

# Heatmap tương quan giữa các biến (không bao gồm Outcome)
plt.figure(figsize=(10, 8))
features_only = df.drop('Outcome', axis=1)
corr_features = features_only.corr(numeric_only=True)

sns.heatmap(corr_features, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('HEATMAP TƯƠNG QUAN GIỮA CÁC BIẾN LÂM SÀNG (KHÔNG BAO GỒM OUTCOME)')
plt.tight_layout()
plt.show()

# Top features tương quan mạnh nhất
plt.figure(figsize=(10, 6))
top_features = corr_target.head(8)
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title('TOP BIẾN TƯƠNG QUAN MẠNH NHẤT VỚI BỆNH TIỂU ĐƯỜNG')
plt.xlabel('Hệ số tương quan (Pearson)')
plt.show()

# Top các cặp biến có tương quan cao nhất (phiên bản đơn giản)
print("\n=== TOP CÁC CẶP BIẾN CÓ TƯƠNG QUAN CAO NHẤT ===")

features_only = df.drop('Outcome', axis=1)
corr_features = features_only.corr(numeric_only=True)

# Lấy các cặp tương quan
corr_pairs = []
for i in range(len(corr_features.columns)):
    for j in range(i+1, len(corr_features.columns)):
        corr_val = corr_features.iloc[i, j]
        corr_pairs.append({
            'Biến 1': corr_features.columns[i],
            'Biến 2': corr_features.columns[j],
            'Tương quan': corr_val,
            'Mức độ': 'Rất cao' if abs(corr_val) > 0.7 else
                     'Cao' if abs(corr_val) > 0.5 else
                     'Trung bình' if abs(corr_val) > 0.3 else 'Thấp'
        })

corr_df = pd.DataFrame(corr_pairs)
corr_df['Tương quan tuyệt đối'] = corr_df['Tương quan'].abs()
top_pairs = corr_df.sort_values('Tương quan tuyệt đối', ascending=False).head(10)

print("🔝 TOP 10 CẶP BIẾN CÓ TƯƠNG QUAN CAO NHẤT:")
for idx, row in top_pairs.iterrows():
    direction = "🟥 DƯƠNG" if row['Tương quan'] > 0 else "🟦 ÂM"
    print(f"{direction} | {row['Biến 1']:20} vs {row['Biến 2']:20} : {row['Tương quan']:7.3f} ({row['Mức độ']})")

# Phân bố các biến quan trọng theo Outcome
important_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, feature in enumerate(important_features):
    row, col = i // 2, i % 2
    sns.boxplot(x='Outcome', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'PHÂN BỐ {feature} THEO OUTCOME')
plt.tight_layout()
plt.show()

# Histogram các biến quan trọng
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, feature in enumerate(important_features):
    row, col = i // 2, i % 2
    df[feature].hist(ax=axes[row, col], bins=20, alpha=0.7, color='skyblue')
    axes[row, col].set_title(f'PHÂN BỐ {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Tần suất')
plt.tight_layout()
plt.show()

# Mối quan hệ Glucose vs BMI
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, alpha=0.7)
plt.title('MỐI QUAN HỆ GIỮA GLUCOSE VÀ BMI')
plt.show()

# Phân bố theo tuổi
plt.figure(figsize=(12, 6))
df.groupby('Age')['Outcome'].mean().plot(kind='bar', color='lightcoral')
plt.title('TỶ LỆ TIỂU ĐƯỜNG THEO TUỔI')
plt.xlabel('Tuổi')
plt.ylabel('Tỷ lệ tiểu đường')
plt.xticks(rotation=45)
plt.show()

# 7️⃣ Nhận xét sơ bộ
print("\n=== NHẬN XÉT SƠ BỘ TỪ EDA ===")
print("📊 PHÁT HIỆN CHÍNH:")
print(f"- Dataset có {df.shape[0]} bệnh nhân, {df.shape[1]-1} features lâm sàng")
print(f"- Tỷ lệ tiểu đường: {outcome_dist[1]*100:.1f}%")
print(f"- Glucose có tương quan mạnh nhất với bệnh tiểu đường: {corr_target['Glucose']:.3f}")
print(f"- BMI và Age cũng là các yếu tố quan trọng")
print(f"- Cần xử lý các giá trị 0 trong Glucose, BloodPressure, etc.")

print("\n🔍 YẾU TỐ NGUY CƠ NỔI BẬT:")
for feature, corr_value in corr_target.items():
    if feature != 'Outcome' and abs(corr_value) > 0.2:
        risk = "TĂNG" if corr_value > 0 else "GIẢM"
        print(f"- {feature}: {risk} nguy cơ (r = {corr_value:.3f})")