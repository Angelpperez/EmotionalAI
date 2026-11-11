import cv2
import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

EMOTION_PROMPTS = [
    "a face showing happiness",
    "a face showing sadness",
    "a face showing anger",
    "a face showing fear",
    "a face showing surprise",
    "a face showing disgust",
    "a neutral face",
]
"""
EMOTION_PROMPTS = [
    "a close-up photo of a person with a big smile, very happy",
    "a close-up photo of a person crying, very sad",
    "a close-up photo of a person with an angry face and furrowed brows",
    "a close-up photo of a scared person with wide open eyes, fear",
    "a close-up photo of a surprised person with mouth open and raised eyebrows",
    "a close-up photo of a disgusted person with a wrinkled nose",
    "a close-up photo of a neutral face with no emotion",
]
    """
EMOTION_LABELS_ES = [
    "feliz",
    "triste",
    "enojado",
    "con miedo",
    "sorprendido",
    "disgustado",
    "neutral",
]

# Cargar CLIP
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

with torch.no_grad():
    text_tokens = clip.tokenize(EMOTION_PROMPTS).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Detector de caras
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # ampliar box
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
            image_features = (
                image_features / image_features.norm(dim=-1, keepdim=True)
            )
            logits = image_features @ text_features.T
            scale = 5.0
            probs = (logits * scale).softmax(dim=-1).cpu().numpy()[0]
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

    cv2.imshow("CLIP", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

