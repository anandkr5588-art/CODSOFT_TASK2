# ==========================================
# CREDIT CARD FRAUD DETECTION (Improved)
# ==========================================
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "fraudTrain.csv")

print("Looking for dataset at:", csv_path)

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Dataset not found!\nExpected location:\n{csv_path}"
    )

df = pd.read_csv(csv_path)

print("="*60)
print(df.head())
print(df.info())
print(df.isnull().sum())

print(df["is_fraud"].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x="is_fraud", data=df)
plt.title("Fraud Distribution")
plt.show()

drop_cols = [
    "trans_date_trans_time","cc_num","first","last","street",
    "city","state","zip","dob","trans_num"
]
df.drop(columns=drop_cols, inplace=True)

le = LabelEncoder()
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = le.fit_transform(df[c])

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print("\n"+"="*60)
    print(name)
    print("Accuracy:", acc)
    print("ROC-AUC :", auc)
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results.append([name, acc, auc])

rf = models["Random Forest"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance)

plt.figure(figsize=(10,6))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis()
plt.title("Random Forest Feature Importance")
plt.show()

print(pd.DataFrame(results, columns=["Model","Accuracy","ROC-AUC"]))

model_path = os.path.join(BASE_DIR, "fraud_detection_model.pkl")
joblib.dump(rf, model_path)
print("Model saved:", model_path)

loaded = joblib.load(model_path)
print("Sample prediction:", loaded.predict(X_test.iloc[[0]])[0])
print("PROJECT COMPLETED")
