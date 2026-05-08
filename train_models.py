import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Try XGBoost
try:
    from xgboost import XGBClassifier
    from collections import Counter
    xgb_available = True
except:
    xgb_available = False


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=["flow_id", "start_time", "end_time"], errors='ignore')

# Handle missing values
df = df.fillna(0)

# Encode labels
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Show label mapping (VERY IMPORTANT for thesis)
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop("Label", axis=1)
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# RANDOM FOREST (BASE MODEL)
# =========================
print("\n=== Random Forest (Base Model) ===")

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# =========================
# CROSS VALIDATION (FIXED)
# =========================
print("\n=== Cross Validation (Random Forest) ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    X, y,
    cv=cv
)

print("Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())


# =========================
# HYPERPARAMETER TUNING
# =========================
print("\n=== Hyperparameter Tuning (Random Forest) ===")

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced"),
    param_grid,
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_rf = grid.best_estimator_

y_pred_best = best_rf.predict(X_test)

print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_best))


# =========================
# FALSE POSITIVE RATE (FIXED)
# =========================
print("\n=== False Positive Rate (FPR) ===")

cm = confusion_matrix(y_test, y_pred_rf)

fpr_list = []

for i in range(len(cm)):
    FP = sum(cm[:, i]) - cm[i, i]
    TN = cm.sum() - (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i])

    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
    fpr_list.append(fpr)

print("FPR per class:", fpr_list)
print("Average FPR:", sum(fpr_list)/len(fpr_list))


# =========================
# XGBOOST (BALANCED)
# =========================
if xgb_available:
    print("\n=== XGBoost (Balanced) ===")

    class_counts = Counter(y_train)
    scale_pos_weight = max(class_counts.values()) / min(class_counts.values())

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        scale_pos_weight=scale_pos_weight
    )

    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

else:
    print("\n⚠ XGBoost not installed. Skipping...")