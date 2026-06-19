import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model(df):

    # Create folder
    os.makedirs("artifacts/models", exist_ok=True)

    # Split features & target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Encoding
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # ✅ SAVE INSIDE FUNCTION
    joblib.dump(model, "artifacts/models/heart_model.pkl")
    joblib.dump(scaler, "artifacts/models/scaler.pkl")
    joblib.dump(X_train.columns.tolist(), "artifacts/models/columns.pkl")

    return model, X_test, y_test