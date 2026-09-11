"""
Trains a Random Forest crop-recommendation model on the standard
Crop_recommendation.csv dataset (N, P, K, temperature, humidity, ph,
rainfall -> crop label) and saves the model + label encoder to disk
for the FastAPI service to load.

Run once locally:
    python train_model.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

df = pd.read_csv(DATA_PATH)

le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

X = df[FEATURES]
y = df["label_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Held-out test accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=le.classes_))

# Feature importance — useful later for the "why this crop" explanation
importances = dict(zip(FEATURES, model.feature_importances_))
print("Feature importances:", importances)

joblib.dump(model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved crop_model.pkl and label_encoder.pkl")
