# pipelinevreed.py
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

print("CWD:", os.getcwd())

# 1) Cargar datos
BASE_DIR = Path(__file__).resolve().parent
print("BASE_DIR:", BASE_DIR)

csv_path = BASE_DIR / "archive" / "04 Eye Tracking Data" / \
           "02 Eye Tracking Data (Features Extracted)" / \
           "EyeTracking_FeaturesExtracted.csv"
df = pd.read_csv(csv_path)
print("Columnas del CSV VREED:")
print(df.columns.tolist())

target_column = "Quad_Cat"

selected_features = [
    "Num_of_Fixations",
    "Mean_Fixation_Duration",
    "SD_Fixation_Duration",
    "Num_of_Saccade",
    "Mean_Saccade_Amplitude",
    "Mean_Saccade_Direction",
]

X = df[selected_features].copy()
y = df[target_column]

X = X.fillna(X.mean(numeric_only=True))

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogReg": LogisticRegression(max_iter=500, n_jobs=-1),
    "SVM_RBF": SVC(kernel="rbf"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    ),
    "MLP_Deep": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42
    )
}

results = []
best_model = None
best_name = None
best_f1 = -np.inf

for name, model in models.items():
    print("=" * 70)
    print(f"Modelo: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 macro: {f1_macro:.3f}")

    cr = classification_report(y_test, y_pred)
    print("\nClassification report:")
    print(cr)

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)

    results.append({"Modelo": name, "Accuracy": acc, "F1_macro": f1_macro})

    with open(f"report_{name}.txt", "w", encoding="utf-8") as f:
        f.write(cr)

    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y))
    disp.plot(values_format="d")
    plt.title(f"Matriz de confusión - {name}")
    plt.tight_layout()
    fig_path = f"cm_{name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Guardada figura:", os.path.abspath(fig_path))

    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_name = name
        best_model = model

results_df = pd.DataFrame(results)
print("\nResumen de modelos:")
print(results_df)

csv_out = "vreed_metricas_modelos.csv"
results_df.to_csv(csv_out, index=False)
print("Guardada tabla:", os.path.abspath(csv_out))

joblib.dump(best_model, f"vreed_best_{best_name}.pkl")
joblib.dump(scaler, "vreed_scaler.pkl")
joblib.dump(selected_features, "vreed_feature_names.pkl")
print(f"\nMejor modelo: {best_name} (F1_macro={best_f1:.3f})")
