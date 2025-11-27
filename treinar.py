import cv2
import numpy as np
import os
from pathlib import Path

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
os.makedirs(TRAINER_DIR, exist_ok=True)

def treinar():
    faces = []
    ids = []
    nomes = {}
    current_id = 0

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for pessoa in os.listdir(DATASET_DIR):
        pasta = os.path.join(DATASET_DIR, pessoa)
        if not os.path.isdir(pasta):
            continue

        nomes[current_id] = pessoa

        for imagem in os.listdir(pasta):
            img_path = os.path.join(pasta, imagem)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces_detectadas = detector.detectMultiScale(img)
            for (x, y, w, h) in faces_detectadas:
                faces.append(img[y:y+h, x:x+w])
                ids.append(current_id)

        current_id += 1

    if len(faces) == 0:
        print("Nenhuma face detectada. Captura mais fotos!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save(os.path.join(TRAINER_DIR, "trainer.yml"))

    # guardar mapeamento de IDs para nomes
    np.save(os.path.join(TRAINER_DIR, "nomes.npy"), nomes)
    print("Treino concluído! Modelo guardado em 'trainer/trainer.yml'.")

if __name__ == "__main__":
    treinar()
