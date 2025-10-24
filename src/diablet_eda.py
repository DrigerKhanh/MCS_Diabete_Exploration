import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("../dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# 1️⃣ Tổng quan dữ liệu
print("=== Tổng quan dữ liệu ===")
print(df.shape)
print(df.info())
print(df.head())

# 2️⃣ Phân bố nhãn bệnh
print("\n=== Phân bố nhãn bệnh (0 = không tiểu đường, 1 = tiểu đường) ===")
print(df['Diabetes_binary'].value_counts(normalize=True))

# 3️⃣ Thống kê mô tả các biến số
print("\n=== Mô tả thống kê ===")
print(df.describe())

# ========================= EDA ============================
# 4️⃣ So sánh trung bình một số yếu tố giữa 2 nhóm
factors = ['BMI', 'Age', 'HighBP', 'HighChol', 'Smoker', 'PhysActivity']
mean_comparison = df.groupby('Diabetes_binary')[factors].mean()
print("\n=== Trung bình các yếu tố theo nhóm bệnh ===")
print(mean_comparison)

# 5️⃣ Tương quan
corr = df.corr(numeric_only=True)
corr_target = corr['Diabetes_binary'].sort_values(ascending=False)

# Hiển thị top 15 biến tương quan mạnh nhất
print("🔍 tương quan của các đặc trưng với Diabetes_binary:")
print(corr_target.head(corr_target.size))

print("Các cột numeric được dùng trong ma trận tương quan:")
print(corr.columns.tolist())
print(f"Tổng cộng: {len(corr.columns)} cột")

# Vẽ biểu đồ trực quan
plt.figure(figsize=(10,6))
sns.barplot(x=corr_target.head(15), y=corr_target.head(15).index, palette="coolwarm")
plt.title('Tương quan giữa các biến và tình trạng tiểu đường')
plt.xlabel('Hệ số tương quan (Pearson)')
plt.ylabel('Biến sức khỏe')
plt.show()

print("\n=== Tương quan giữa các biến và bệnh tiểu đường ===")
print(corr_target.head(15))

# Vẽ heatmap tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Heatmap tương quan giữa các biến sức khỏe")
plt.show()

# 6️⃣ Một số biểu đồ trực quan
plt.figure(figsize=(8, 5))
sns.boxplot(x='Diabetes_binary', y='BMI', data=df)
plt.title("Phân bố BMI theo tình trạng tiểu đường")
plt.show()

# Thêm phân bố của các biến số quan trọng
important_vars = ['BMI', 'Age', 'GenHlth', 'PhysHlth']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, var in enumerate(important_vars):
    df[var].hist(ax=axes[i//2, i%2], bins=20, alpha=0.7)
    axes[i//2, i%2].set_title(f'Phân bố {var}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='HighBP', y='Diabetes_binary', data=df)
plt.title("Tỷ lệ mắc bệnh theo huyết áp cao")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='HighChol', y='Diabetes_binary', data=df)
plt.title("Tỷ lệ mắc bệnh theo cholesterol cao")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='PhysActivity', y='Diabetes_binary', data=df)
plt.title("Tỷ lệ mắc bệnh theo hoạt động thể chất")
plt.show()

# Tập trung vào top 10 biến quan trọng nhất
top_10_features = corr_target.head(11).index[1:]  # Bỏ chính Diabetes_binary

plt.figure(figsize=(10, 8))
sns.heatmap(df[top_10_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Tương quan giữa các yếu tố nguy cơ chính')
plt.show()

# Tỷ lệ tiểu đường theo nhóm tuổi
plt.figure(figsize=(12, 6))
df.groupby('Age')['Diabetes_binary'].mean().plot(kind='bar')
plt.title('Tỷ lệ tiểu đường theo nhóm tuổi')
plt.ylabel('Tỷ lệ tiểu đường')
plt.show()

# Theo giới tính (nếu có biến Sex)
if 'Sex' in df.columns:
    pd.crosstab(df['Sex'], df['Diabetes_binary'], normalize='index').plot(kind='bar')
    plt.title('Tỷ lệ tiểu đường theo giới tính')
    plt.show()

# 7️⃣ Nhận xét sơ bộ
print("\n=== Nhận xét sơ bộ từ EDA ===")
print("- Nhóm bị tiểu đường có BMI và tuổi trung bình cao hơn rõ rệt.")
print("- Huyết áp cao (HighBP) và cholesterol cao (HighChol) là hai yếu tố tương quan mạnh nhất.")
print("- Hoạt động thể chất giúp giảm nguy cơ mắc bệnh.")
print("- Hút thuốc và vận động thể chất cũng có ảnh hưởng nhẹ.")

