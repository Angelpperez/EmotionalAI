# VREED.py
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
import cv2

def train_vreed_models():
    BASE_DIR = Path(__file__).resolve().parent
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

    # NaNs -> media
    X = X.fillna(X.mean(numeric_only=True))

    # Escalado
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

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
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {acc:.3f}")
        print(f"F1 macro: {f1_macro:.3f}")
        cr = classification_report(y_test, y_pred)
        print("\nClassification report:")
        print(cr)

        cm = confusion_matrix(y_test, y_pred)
        print("Matriz de confusión:")
        print(cm)


        results.append({
            "Modelo": name,
            "Accuracy": acc,
            "F1_macro": f1_macro
        })


        with open(f"report_{name}.txt", "w", encoding="utf-8") as f:
            f.write(cr)

        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y))
        disp.plot(values_format="d")
        plt.title(f"Matriz de confusión {name}")
        plt.tight_layout()
        plt.savefig(f"cm_{name}.png", dpi=300)
        plt.close()

        # elegir mejor modelo por F1
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results)
    print("\nResumen de modelos:")
    print(results_df)
    results_df.to_csv("vreed_metricas_modelos.csv", index=False)

    joblib.dump(best_model, f"vreed_best_{best_name}.pkl")
    joblib.dump(scaler, "vreed_scaler.pkl")
    joblib.dump(selected_features, "vreed_feature_names.pkl")
    print(f"\nMejor modelo: {best_name} (F1_macro={best_f1:.3f})")

    return best_model, scaler, selected_features


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_pupil_in_eye(eye_gray):
    """
    Devuelve:
      center: (x,y)
      radius: radio aproximado
      edges: bordes Canny
      seg: máscara binaria de pupila
      preproc: imagen preprocesada (mediana + gauss)
    """
    eye_eq = cv2.equalizeHist(eye_gray)

    med = cv2.medianBlur(eye_eq, 7)

    blur_soft = cv2.GaussianBlur(eye_eq, (0, 0), 1)
    sharp = cv2.addWeighted(med, 1.2, blur_soft, -0.5, 0)

    edges = cv2.Canny(blur_soft, 100, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    inv = 255 - opened
    _, seg = cv2.threshold(
        inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, edges, seg, sharp

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 40 or len(contour) < 5:
        return None, None, edges, seg, sharp

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (MA, ma), _ = ellipse
    radius = int((MA + ma) / 4)

    return (int(cx), int(cy)), radius, edges, seg, sharp


def compute_features_from_angles(angles, fps=30.0):
    """
    angles: lista de ángulos de mirada (grados) en una ventana.
    Devuelve un vector en el orden:
      [Num_of_Fixations, Mean_Fixation_Duration, SD_Fixation_Duration,
       Num_of_Saccade,   Mean_Saccade_Amplitude, Mean_Saccade_Direction]
    """
    angles = np.asarray(angles, dtype=float)
    if angles.size < 2:
        return [0, 0.0, 0.0, 0, 0.0, 0.0]

    diffs = np.diff(angles)

    # Umbrales 
    TH_FIX = 2.0   
    TH_SACC = 5.0  

    fix_durations = []
    current_len = 1

    for d in diffs:
        if abs(d) <= TH_FIX:
            current_len += 1
        else:
            if current_len > 1:
                fix_durations.append(current_len / fps)
            current_len = 1

    if current_len > 1:
        fix_durations.append(current_len / fps)

    num_fix = len(fix_durations)
    if num_fix > 0:
        mean_fix = float(np.mean(fix_durations))
        sd_fix = float(np.std(fix_durations))
    else:
        mean_fix = 0.0
        sd_fix = 0.0

    sacc_mask = np.abs(diffs) > TH_SACC
    sacc_amps = np.abs(diffs[sacc_mask])
    sacc_dirs = np.sign(diffs[sacc_mask])

    num_sacc = int(sacc_amps.size)
    if num_sacc > 0:
        mean_amp = float(np.mean(sacc_amps))
        mean_dir = float(np.mean(sacc_dirs))
    else:
        mean_amp = 0.0
        mean_dir = 0.0

    return [num_fix, mean_fix, sd_fix, num_sacc, mean_amp, mean_dir]

def run_webcam_with_emotion(model, scaler, feature_names, fps=30.0):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    WINDOW_FRAMES = int(fps * 2)   # ventana de tiempo para 2s
    angle_buffer = []

    quad_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    current_quad = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_full = np.zeros_like(frame)
        seg_full = np.zeros_like(frame)
        pre_full = np.zeros_like(frame)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        frame_angle = None  # angulo promedio de ambos ojos 

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]

            eyes_raw = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 20)
            )

            eye_candidates = []
            for (ex, ey, ew, eh) in eyes_raw:
                cy_rel = ey + eh / 2
                if cy_rel > 0.6 * h:
                    continue
                ratio = ew / float(eh)
                if ratio < 0.7 or ratio > 3.0:
                    continue
                eye_candidates.append((ex, ey, ew, eh))

            eye_candidates = sorted(eye_candidates, key=lambda e: -e[2])[:2]
            eye_candidates = sorted(eye_candidates, key=lambda e: e[0])

            angles_this_frame = []

            for idx, (ex, ey, ew, eh) in enumerate(eye_candidates):
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]

                center, radius, edges, seg, preproc = detect_pupil_in_eye(eye_gray)

                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                seg_color = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
                pre_color = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR)

                edges_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = edges_color
                seg_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = seg_color
                pre_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = pre_color

                if center is None:
                    continue

                if radius < 3 or radius > min(ew, eh) // 2:
                    continue

                cx_global = x + ex + center[0]
                cy_global = y + ey + center[1]
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2

                side = max(ew, eh)
                half = side // 2
                tl_x = eye_center_x - half
                tl_y = eye_center_y - half
                br_x = eye_center_x + half
                br_y = eye_center_y + half

                h_frame, w_frame = frame.shape[:2]
                tl_x = max(0, tl_x)
                tl_y = max(0, tl_y)
                br_x = min(w_frame - 1, br_x)
                br_y = min(h_frame - 1, br_y)

                cv2.rectangle(
                    frame,
                    (tl_x, tl_y),
                    (br_x, br_y),
                    (0, 255, 0),
                    2
                )

                label = "L" if idx == 0 else "R"
                cv2.putText(
                    frame, label,
                    (tl_x + 5, tl_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2
                )

                cv2.circle(
                    frame,
                    (cx_global, cy_global),
                    3,
                    (0, 255, 0), -1
                )

                vx = cx_global - eye_center_x
                vy = cy_global - eye_center_y
                norm = np.hypot(vx, vy) + 1e-6

                L = 80
                x2 = int(eye_center_x + L * vx / norm)
                y2 = int(eye_center_y + L * vy / norm)

                cv2.line(
                    frame,
                    (eye_center_x, eye_center_y),
                    (x2, y2),
                    (0, 255, 255), 2
                )

                angle_deg = np.degrees(np.arctan2(-vy, vx))
                angles_this_frame.append(angle_deg)
                text = f"{angle_deg:+.1f}"
                cv2.putText(
                    frame, text,
                    (eye_center_x + 5, eye_center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1
                )

            if angles_this_frame:
                frame_angle = float(np.mean(angles_this_frame))

        if frame_angle is not None:
            angle_buffer.append(frame_angle)
            if len(angle_buffer) > WINDOW_FRAMES:
                angle_buffer = angle_buffer[-WINDOW_FRAMES:]

        if len(angle_buffer) == WINDOW_FRAMES:
            feat_vec = compute_features_from_angles(angle_buffer, fps=fps)
            X_live = np.array(feat_vec, dtype=float).reshape(1, -1)
            X_live_scaled = scaler.transform(X_live)
            quad_pred = int(model.predict(X_live_scaled)[0])
            current_quad = quad_pred
            quad_counts[quad_pred] = quad_counts.get(quad_pred, 0) + 1

        if current_quad is not None:
            cv2.putText(
                frame,
                f"Emotion Quad_Cat: {current_quad}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2
            )

        cv2.imshow("Completa", frame)
        cv2.imshow("Bordes con Canny", edges_full)
        cv2.imshow("Segmentación binaria", seg_full)
        cv2.imshow("Preprocesamiento MEDIANA Y GAUSS", pre_full)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---- histograma de cuadrantes predichos ----
    labels = sorted(quad_counts.keys())
    values = [quad_counts[k] for k in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Quad_Cat predicho")
    plt.ylabel("Frecuencia según ventana")
    plt.title("Distribución de estados emocionales predichos")
    plt.tight_layout()
    plt.savefig("hist_emociones_webcam.png", dpi=300)
    plt.close()


def main():
    #entrenar modelos VREED
    best_model, scaler, feat_names = train_vreed_models()

    #thread para webcam en tiempo de ejecución
    run_webcam_with_emotion(best_model, scaler, feat_names, fps=30.0)


if __name__ == "__main__":
    main()

