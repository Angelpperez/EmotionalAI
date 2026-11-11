# vreed_models.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pathlib import Path

# 1. Cargar datos VREED (eye tracking)
BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "archive" / "04 Eye Tracking Data" / \
           "02 Eye Tracking Data (Features Extracted)" / \
           "EyeTracking_FeaturesExtracted.csv"
df = pd.read_csv(csv_path)
print(df.columns.tolist())

# 2. Elegir la variable objetivo
# En este CSV solo está 'Quad_Cat' (4 clases).
# Si luego quieres Valence/Arousal, hay que unir con los cuestionarios.
target_column = 'Quad_Cat'

# 3. Separar X, y
y = df[target_column]
X = df.drop(columns=[target_column])

# Si quieres quitar también otras etiquetas, descomenta (cuando existan):
# X = X.drop(columns=['Quad_Cat','Arousal','Valence'], errors='ignore')
# y = df[target_column]

# 4. Manejo simple de NaNs (promedio por columna)
X = X.fillna(X.mean(numeric_only=True))

# 5. Escalado robusto
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train/test split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Definir modelos a comparar
models = {
    "LogReg": LogisticRegression(max_iter=500, n_jobs=-1),
    "SVM_RBF": SVC(kernel='rbf'),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    ),
    "MLP_Deep": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=500,
        random_state=42
    )
}

# 8. Entrenar y evaluar
for name, model in models.items():
    print("="*70)
    print(f"Modelo: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 macro: {f1_macro:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
