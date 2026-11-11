import cv2
import torch
import clip
from PIL import Image
import numpy as np
import time

# ----------------- CONFIG -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", device)

EMOTION_PROMPTS = [
    "a face showing happiness",
    "a face showing sadness",
    "a face showing anger",
    "a face showing fear",
    "a face showing surprise",
    "a face showing disgust",
    "a neutral face",
]

EMOTION_LABELS_ES = [
    "feliz",
    "triste",
    "enojado",
    "con miedo",
    "sorprendido",
    "disgustado",
    "neutral",
]

# ------------- CARGAR CLIP ---------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

with torch.no_grad():
    text_tokens = clip.tokenize(EMOTION_PROMPTS).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# --------- DETECTOR DE CARAS -------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------- CAPTURA DE VIDEO --------------
cap = cv2.VideoCapture(0)  # misma cámara que usas en el clásico
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam")

times = []
total_frames = 0
frames_con_cara = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    start = time.perf_counter()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        frames_con_cara += 1

        pad = int(0.1 * h)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)

        face_img = frame[y0:y1, x0:x1]
        if face_img.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        best_idx = int(np.argmax(probs))
        emotion_es = EMOTION_LABELS_ES[best_idx]
        best_score = probs[best_idx]

        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        text = f"{emotion_es}: {best_score:.2f}"
        cv2.putText(
            frame,
            text,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    end = time.perf_counter()
    times.append(end - start)

    cv2.imshow("CLIP (baseline deep)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ------------- MÉTRICAS DE RENDIMIENTO -------------
total_time = sum(times)
N = len(times)
fps = N / total_time if total_time > 0 else 0.0

print("\n=== Métricas CLIP ===")
print(f"Frames procesados: {N}")
print(f"Frames totales capturados: {total_frames}")
print(f"Frames con al menos una cara: {frames_con_cara}")
print(f"Tiempo total procesamiento: {total_time:.2f} s")
print(f"FPS efectivos (solo CLIP): {fps:.2f}")
