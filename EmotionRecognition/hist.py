import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_histogramas():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("Presiona ESPACIO para capturar un frame, q para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Camara", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # --- Preprocesamientos ---
            eye_eq   = cv2.equalizeHist(gray)
            mediana  = cv2.medianBlur(eye_eq, 9)
            gauss    = cv2.GaussianBlur(eye_eq, (9, 9), 1)

            # --- Histogramas ---
            imgs = [eye_eq, mediana, gauss]
            titulos = ["Ecualizada", "Mediana 9x9", "Gauss 9x9"]

            plt.figure(figsize=(10,4))

            for i, img in enumerate(imgs):
                plt.subplot(1,3,i+1)
                plt.hist(img.ravel(), bins=256, range=(0,256), color='gray')
                plt.title(titulos[i])
                plt.xlabel("Nivel de gris")
                plt.ylabel("Frecuencia")

            plt.tight_layout()
            plt.show()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mostrar_histogramas()
