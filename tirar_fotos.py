import cv2
import os

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def capturar_fotos():
    nome = input("Nome da pessoa: ").strip()
    pasta_pessoa = os.path.join(DATASET_DIR, nome)
    os.makedirs(pasta_pessoa, exist_ok=True)

    cap = cv2.VideoCapture(0)
    contador = 0

    print("A capturar fotos... Prima 'q' para parar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Camera - Captura", frame)

        foto_path = os.path.join(pasta_pessoa, f"{contador}.jpg")
        cv2.imwrite(foto_path, frame)
        contador += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{contador} fotos guardadas para {nome}")

if __name__ == "__main__":
    capturar_fotos()
