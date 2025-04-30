import cv2
import face_recognition
import threading
import numpy as np
import requests
import time
from datetime import datetime, time as dtime
from dao.mysql import DatabaseConnection
from PIL import Image
from io import BytesIO
#from utils.myserial import MYSerial

class FaceDetectionRecognition:
    def __init__(self):
        # Configurações de exibição
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cadastrado_cor = (0, 255, 0)
        self.desconhecido_cor = (0, 0, 255)
        self.tamanho = 1
        self.espessura = 2
        self.font_scale = 1.0
        self.thickness = 2

        # Pula frames para economizar processamento
        self.frame_skip = 5
        # Fator de redução de resolução para detecção
        self.resize_scale = 0.5
        # Limiar de distância para reconhecimento
        self.min_distance = 0.6

        # Inicialização da câmera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(3, 320)
        self.cap.set(4, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Listas de rostos conhecidos e metadados
        self.known_faces = []
        self.known_names = []
        self.know_phones = []
        self.know_matricula = []

        # Sessões usadas para evitar reenvio
        self.used_sessions = {}
        self.session = requests.Session()
        #self.url = "http://127.0.0.1:8000/send_message"

        # Janelas de horário
        self.start_time_enter = dtime(8, 00) # Horário de entrada
        self.end_time_enter = dtime(12, 00) # Horário de Said
        self.start_time_out = dtime(13, 00)
        self.end_time_out = dtime(22, 0)

        # Serial
        # self.my_serial = MYSerial('COM7', 9600)

        # Carrega rostos e informações do banco
        db = DatabaseConnection(
            dbname="image_db", user="ifba", password="ifba6514", host="localhost", port="3306"
        )
        records = db.get_all()
        for record in records:
            # Desempacota registro (matricula, nome, telefone, _, imagem)
            matricula, name, phone, _, image_binary = record
            img_bytes = bytearray(image_binary)
            img_pil = Image.open(BytesIO(img_bytes))
            img = np.array(img_pil)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Codificação do rosto
            enc = face_recognition.face_encodings(img)[0]
            self.known_faces.append(enc)
            self.known_names.append(name)
            self.know_phones.append(phone)
            self.know_matricula.append(matricula)

    def send_message_async(self, payload):
        def task(data):
            try:
                r = self.session.post(self.url, json=data, timeout=3)
                if r.status_code == 200:
                    print("Mensagem enviada com sucesso.", r.json())
                else:
                    print("Erro ao enviar mensagem:", r.status_code, r.text)
            except Exception as e:
                print("Exception no envio:", e)

        threading.Thread(target=task, args=(payload,), daemon=True).start()

    def get_payload(self, name, phone):
        now = datetime.now().time()
        if name in self.used_sessions:
            return None  # Já enviada

        if self.start_time_enter <= now <= self.end_time_enter:
            on_school = False
        elif self.start_time_out <= now <= self.end_time_out:
            on_school = True
        else:
            return f"Fora do horário permitido para enviar a mensagem."

        self.used_sessions[name] = True
        return {"name": name, "phone": phone, "on_school": on_school}

    def detect_recognize_faces(self):
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("test")
                break
            frame_count += 1
            # Pula frames para aliviar CPU
            if frame_count % self.frame_skip != 0:
                cv2.imshow('img', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Reduz resolução e converte para RGB
            small = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Detecção em baixa resolução usando HOG (mais rápido)
            locs = face_recognition.face_locations(rgb_small, model='hog')
            encs = face_recognition.face_encodings(rgb_small, locs)

            for (top, right, bottom, left), enc in zip(locs, encs):
                # Redimensiona coordenadas de volta ao frame original
                top, right, bottom, left = [int(v / self.resize_scale) for v in (top, right, bottom, left)]

                # Verificação por distância mínima
                distances = face_recognition.face_distance(self.known_faces, enc)
                idx = np.argmin(distances)
                if distances[idx] < self.min_distance:
                    name = self.known_names[idx]
                    phone = self.know_phones[idx]
                    color = self.cadastrado_cor
                    serial_data = 1
                    payload = self.get_payload(name, phone)
                    if isinstance(payload, dict):
                        self.send_message_async(payload)
                    elif isinstance(payload, str):
                        print(payload)
                else:
                    name = "Not Found"
                    color = self.desconhecido_cor
                    serial_data = 0

                # Envia sinal serial
                #self.my_serial.receive(serial_data)
                # Desenha no frame
                cv2.rectangle(frame, (left, top), (right, bottom), color, self.espessura)
                cv2.putText(frame, name, (left, top - 10), self.font, self.tamanho, color, self.espessura)

            cv2.imshow('img', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Limita FPS para poupar CPU
            time.sleep(0.01)

        self.cap.release()
        cv2.destroyAllWindows()
        self.session.close()

    def start(self):
        t = threading.Thread(target=self.detect_recognize_faces, daemon=True)
        t.start()
        t.join()

if __name__ == "__main__":
    FaceDetectionRecognition().start()
