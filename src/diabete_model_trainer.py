import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def train_and_save_model():
    """Train model once and save for later use"""
    print("=== TRAINING DIABETES PREDICTION MODEL ===")

    # 1. Load và tiền xử lý
    df = pd.read_csv("../dataset/diabetes.csv")

    # Xử lý missing values
    df_processed = df.copy()
    clinical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for feature in clinical_features:
        df_processed[feature] = df_processed[feature].replace(0, np.nan)

    for feature in clinical_features:
        df_processed[feature] = df_processed.groupby('Outcome')[feature].transform(
            lambda x: x.fillna(x.median())
        )

    # 2. Chọn features và chuẩn hóa
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df_processed[feature_names]
    y = df_processed['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)

    # 4. Đánh giá
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Model Performance:")
    print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

    # In classification report chi tiết
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # 5. Vẽ biểu đồ ROC-AUC
    plot_roc_auc(y_test, y_pred_proba)

    # 6. Vẽ Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)

    # 7. Lưu model và scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'performance': {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    }

    joblib.dump(model_data, '../resource/model/diabetes_model.pkl')
    print("✅ Model trained and saved successfully!")

    return model_data


def plot_roc_auc(y_test, y_pred_proba):
    """Vẽ biểu đồ ROC-AUC Curve"""
    # Tính toán ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    # Tùy chỉnh biểu đồ
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC-AUC Curve - Diabetes Prediction Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Thêm thông tin performance
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # In một số thresholds quan trọng
    print(f"📈 ROC-AUC Score: {roc_auc:.3f}")


def plot_confusion_matrix(y_test, y_pred):
    """Vẽ Confusion Matrix chi tiết"""
    # Tính confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])

    # Tùy chỉnh biểu đồ
    plt.title('Confusion Matrix - Diabetes Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Thêm thông tin chi tiết
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Hiển thị các metrics
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.text(2.3, 0.5, metrics_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # In kết quả chi tiết
    print("\n🔍 Confusion Matrix Details:")
    print(f"True Negatives (TN): {tn} - Correctly predicted no diabetes")
    print(f"False Positives (FP): {fp} - Incorrectly predicted diabetes")
    print(f"False Negatives (FN): {fn} - Missed diabetes cases")
    print(f"True Positives (TP): {tp} - Correctly predicted diabetes")
    print(f"\n📊 Additional Metrics:")
    print(f"Precision: {precision:.3f} - When model says diabetes, how often is it correct")
    print(f"Recall: {recall:.3f} - What proportion of actual diabetes cases were identified")
    print(f"F1-Score: {f1:.3f} - Balance between precision and recall")


if __name__ == "__main__":
    # Chỉ chạy 1 lần để train model
    train_and_save_model()