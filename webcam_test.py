import cv2

def verifica_webcam(indice=0, timeout_frames=30):
    """
    Tenta abrir a webcam (padrão = 0), lê alguns frames e retorna True se
    conseguir capturar pelo menos um frame com sucesso.
    """
    cap = cv2.VideoCapture(f'/dev/video{indice}', cv2.CAP_V4L2)  # em Windows, ajuda a evitar warnings
    if not cap.isOpened():
        print(f"❌ Não foi possível abrir a webcam de índice {indice}.")
        return False

    # opcional: configure resolução mínima para teste
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    got_frame = False
    for _ in range(timeout_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        got_frame = True
        break

    cap.release()
    if got_frame:
        print("✅ Webcam funcionando corretamente!")
    else:
        print("❌ Webcam abriu, mas não capturou nenhum frame.")
    return got_frame

if __name__ == "__main__":
    # Executa o teste
    funcionou = verifica_webcam()
