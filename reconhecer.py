import cv2
import numpy as np
import os

TRAINER_DIR = "trainer"

def reconhecer_multiplas():
    if not os.path.exists(os.path.join(TRAINER_DIR, "trainer.yml")):
        print("Treinar primeiro!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(TRAINER_DIR, "trainer.yml"))
    nomes = np.load(os.path.join(TRAINER_DIR, "nomes.npy"), allow_pickle=True).item()

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    print("A reconhecer múltiplas pessoas... Prima 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face_gray)

            # Ajustar threshold para reconhecer mais pessoas
            if conf < 100:
                nome = nomes[id_]
            else:
                nome = "Desconhecido"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, nome, (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento Facial - Multiplas Pessoas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    reconhecer_multiplas()
