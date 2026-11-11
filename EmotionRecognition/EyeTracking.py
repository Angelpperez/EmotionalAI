
import cv2
from fer.fer import FER

detector = FER(mtcnn=True)  
cap = cv2.VideoCapture(0)   
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = detector.detect_emotions(frame)

    for r in results:
        (x, y, w, h) = r["box"]
        emotions = r["emotions"]
        emotion, score = max(emotions.items(), key=lambda t: t[1])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{emotion}: {score:.2f}"
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Emociones", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

