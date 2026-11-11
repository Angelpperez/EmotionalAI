import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

metrics = {
    "L": {
        "frames": 0,
        "valid_det": 0,
        "sum_R": 0.0,
        "sum_var": 0.0,
        "sum_H": 0.0,
        "sum_edge_density": 0.0,
        "sum_contour_len": 0.0,
        "sum_RA": 0.0,
        "cx": [],
        "cy": [],
        "r": [],
        "theta": [],
        "angle_sign_changes": 0,
        "prev_sign": None,
    },
    "R": {
        "frames": 0,
        "valid_det": 0,
        "sum_R": 0.0,
        "sum_var": 0.0,
        "sum_H": 0.0,
        "sum_edge_density": 0.0,
        "sum_contour_len": 0.0,
        "sum_RA": 0.0,
        "cx": [],
        "cy": [],
        "r": [],
        "theta": [],
        "angle_sign_changes": 0,
        "prev_sign": None,
    },
}


def detect_pupil_in_eye(eye_gray):
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


def update_metrics(label, preproc, edges, seg, ew, eh,
                   cx_global=None, cy_global=None,
                   radius=None, angle_deg=None):
    m = metrics[label]
    roi_area = float(ew * eh)

    m["frames"] += 1

    pre_f = preproc.astype(np.float32)
    R = float(pre_f.max() - pre_f.min())
    var = float(pre_f.var())

    hist = cv2.calcHist([preproc], [0], None, [256], [0, 256])
    p = hist / roi_area
    p = p[p > 0]
    H = float(-np.sum(p * np.log2(p)))

    edge_density = float(np.count_nonzero(edges)) / roi_area

    cont_edges, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if cont_edges:
        lengths = [len(c) for c in cont_edges]
        mean_len = float(np.mean(lengths))
    else:
        mean_len = 0.0

    seg_area = float(np.count_nonzero(seg))
    RA = seg_area / roi_area

    m["sum_R"] += R
    m["sum_var"] += var
    m["sum_H"] += H
    m["sum_edge_density"] += edge_density
    m["sum_contour_len"] += mean_len
    m["sum_RA"] += RA

    if cx_global is not None and cy_global is not None and radius is not None and angle_deg is not None:
        m["valid_det"] += 1
        m["cx"].append(cx_global)
        m["cy"].append(cy_global)
        m["r"].append(radius)
        m["theta"].append(angle_deg)

        sign = np.sign(angle_deg)
        if m["prev_sign"] is not None and sign != 0:
            if np.sign(m["prev_sign"]) != np.sign(sign):
                m["angle_sign_changes"] += 1
        m["prev_sign"] = angle_deg


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

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

            for idx, (ex, ey, ew, eh) in enumerate(eye_candidates):
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]

                center, radius, edges, seg, preproc = detect_pupil_in_eye(eye_gray)

                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                seg_color = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
                pre_color = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR)

                edges_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = edges_color
                seg_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = seg_color
                pre_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = pre_color

                label = "L" if idx == 0 else "R"

                if center is None:
                    update_metrics(label, preproc, edges, seg, ew, eh)
                    continue

                if radius < 3 or radius > min(ew, eh) // 2:
                    update_metrics(label, preproc, edges, seg, ew, eh)
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

                label_text = "L" if idx == 0 else "R"
                cv2.putText(
                    frame, label_text,
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

                angle_deg = float(np.degrees(np.arctan2(-vy, vx)))
                text = f"{angle_deg:+.1f}"
                cv2.putText(
                    frame, text,
                    (eye_center_x + 5, eye_center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1
                )

                update_metrics(label, preproc, edges, seg, ew, eh,
                               cx_global, cy_global, radius, angle_deg)

        cv2.imshow("Completa", frame)
        cv2.imshow("Bordes (Canny)", edges_full)
        cv2.imshow("Segmentacion Pupila", seg_full)
        cv2.imshow("Preprocesamiento (Mediana + Sharpen)", pre_full)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    for label in ["L", "R"]:
        m = metrics[label]
        if m["frames"] == 0:
            continue
        print(f"\n=== Métricas ojo {label} ===")
        print(f"Frames con ROI de ojo: {m['frames']}")
        print(f"Detecciones válidas pupila: {m['valid_det']} "
              f"({m['valid_det']/m['frames']*100:.2f}%)")

        f = float(m["frames"])
        print(f"Rango dinámico medio R: {m['sum_R']/f:.2f}")
        print(f"Varianza media: {m['sum_var']/f:.2f}")
        print(f"Entropía media H: {m['sum_H']/f:.3f}")
        print(f"Densidad de borde media: {m['sum_edge_density']/f:.4f}")
        print(f"Longitud media de contornos: {m['sum_contour_len']/f:.2f}")
        print(f"Relación área máscara/ROI (RA) media: {m['sum_RA']/f:.4f}")

        if m["cx"]:
            print(f"Desv. estándar cx: {np.std(m['cx']):.2f}")
            print(f"Desv. estándar cy: {np.std(m['cy']):.2f}")
            print(f"Desv. estándar radio r: {np.std(m['r']):.2f}")
        if m["theta"]:
            print(f"Desv. estándar ángulo θ: {np.std(m['theta']):.2f}")
            print(f"Índice de actividad ocular "
                  f"(cambios de signo por frame): "
                  f"{m['angle_sign_changes']/len(m['theta']):.4f}")


if __name__ == "__main__":
    main()
