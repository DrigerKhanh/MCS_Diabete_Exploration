import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib


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

    # 5. Lưu model và scaler
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


if __name__ == "__main__":
    # Chỉ chạy 1 lần để train model
    train_and_save_model()