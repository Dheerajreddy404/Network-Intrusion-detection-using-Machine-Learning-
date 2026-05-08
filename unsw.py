import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("unsw.csv")

# =========================
# SELECT FEATURES (SIMPLE)
# =========================
features = ["sbytes", "dbytes", "spkts", "dpkts", "dur"]

df = df[features + ["label"]]

df = df.fillna(0)

X = df[features]
y = df["label"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL
# =========================
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rf.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = rf.predict(X_test)

# =========================
# RESULTS
# =========================
print("\n=== UNSW RESULTS ===")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))