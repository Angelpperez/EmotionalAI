import cv2
import numpy as np


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


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("DEBUG No se puede abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_full = np.zeros_like(frame)      
        seg_full   = np.zeros_like(frame)      
        pre_full   = np.zeros_like(frame)      

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            roi_gray  = gray[y:y + h, x:x + w]


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

                edges_color = cv2.cvtColor(edges,   cv2.COLOR_GRAY2BGR)
                seg_color   = cv2.cvtColor(seg,     cv2.COLOR_GRAY2BGR)
                pre_color   = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR)

                edges_full[y + ey:y + ey + eh, x + ex:x + ex + ew] = edges_color
                seg_full[y + ey:y + ey + eh,   x + ex:x + ex + ew] = seg_color
                pre_full[y + ey:y + ey + eh,   x + ex:x + ex + ew] = pre_color

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

                # línea amarilla
                cv2.line(
                    frame,
                    (eye_center_x, eye_center_y),
                    (x2, y2),
                    (0, 255, 255), 2
                )

                # ángulo en grados 
                angle_deg = np.degrees(np.arctan2(-vy, vx))
                text = f"{angle_deg:+.1f}"
                cv2.putText(
                    frame, text,
                    (eye_center_x + 5, eye_center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1
                )

        cv2.imshow("Completa", frame)
        cv2.imshow("Bordes Canny", edges_full)
        cv2.imshow("Segmentacion", seg_full)
        cv2.imshow("Preprocesamiento MEDIANA Y GAUSS", pre_full)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
