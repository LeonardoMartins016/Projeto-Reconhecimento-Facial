import cv2
import numpy as np
import os
from datetime import datetime
import threading
import queue
import time
import dlib
import json
from pathlib import Path
from scipy.spatial import distance as dist # Adicionar esta importação

# Configuração dos diretórios
KNOWN_FACES_DIR = "know_faces"
ENCODINGS_FILE = "face_encodings.json"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

class FaceRecognitionLogin:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_names = []
        self.process_this_frame = True
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing = False
        self.is_logged_in = False
        self.current_user = None
        
        # Carrega os modelos específicos
        print("Carregando modelos de reconhecimento facial...")
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        print("Modelos carregados com sucesso!")
        
        # Carrega os encodings salvos
        self.load_known_faces()

        # Constantes para verificação de vivacidade (EAR)
        self.EYE_AR_LIVENESS_THRESH = 0.20 # Limiar de EAR para considerar o olho aberto. Ajuste conforme necessário.

    def _calculate_ear_from_points(self, eye_pts):
        """Calcula a Proporção de Aspecto do Olho (EAR) a partir dos pontos do olho."""
        # P0, P1, P2, P3, P4, P5 (índices 0 a 5 para os 6 pontos do olho)
        #    P1  P2
        # P0        P3
        #    P5  P4
        try:
            A = dist.euclidean(eye_pts[1], eye_pts[5]) # Distância vertical |P1-P5|
            B = dist.euclidean(eye_pts[2], eye_pts[4]) # Distância vertical |P2-P4|
            C = dist.euclidean(eye_pts[0], eye_pts[3]) # Distância horizontal |P0-P3|

            if C == 0: # Evitar divisão por zero
                return 0.3 # Retorna um valor de olho aberto padrão se C for 0 (improvável)
            ear = (A + B) / (2.0 * C)
            return ear
        except Exception:
            return 0.3 # Em caso de erro no cálculo, assume olho aberto para não bloquear indevidamente

    def _check_liveness_simple(self, shape):
        """
        Verificação de vivacidade simples baseada na abertura dos olhos.
        Retorna True se os olhos parecerem abertos, False caso contrário.
        """
        try:
            # Marcos para o olho esquerdo: pontos 36-41 (índices 0-5 após extração)
            # Marcos para o olho direito: pontos 42-47 (índices 0-5 após extração)
            
            left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

            left_ear = self._calculate_ear_from_points(left_eye_pts)
            right_ear = self._calculate_ear_from_points(right_eye_pts)

            ear = (left_ear + right_ear) / 2.0

            if ear > self.EYE_AR_LIVENESS_THRESH:
                return True # Olhos parecem abertos
            else:
                # print(f"Liveness check failed: EAR {ear:.2f}") # Para debug
                return False # Olhos parecem fechados ou parcialmente fechados
        except Exception as e:
            # print(f"Error in liveness check: {e}") # Para debug
            return True # Em caso de erro na detecção dos landmarks, permite o login para não bloquear indevidamente.
                        # Poderia ser False para maior segurança, mas arrisca falsos negativos.

    def save_face_encoding(self, name, face_encoding):
        """Salva o encoding facial no arquivo JSON"""
        encodings = {}
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'r') as f:
                encodings = json.load(f)
        
        if name not in encodings:
            encodings[name] = []
        
        # Converte o encoding para lista para serialização JSON
        encodings[name].append(face_encoding.tolist())
        
        with open(ENCODINGS_FILE, 'w') as f:
            json.dump(encodings, f)

    def setup_camera(self):
        """Configura e retorna a webcam com configurações otimizadas"""
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise Exception("Não foi possível acessar a webcam")
        
        # Configurações otimizadas para performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduz buffer
        return video_capture

    def process_frame_for_recognition(self, frame, scale=0.25):
        """Processa o frame para detecção facial com escala reduzida"""
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    def face_recognition_worker(self):
        """Worker thread para processamento de reconhecimento facial"""
        while True:
            try:
                if not self.frame_queue.empty():
                    rgb_small_frame = self.frame_queue.get(timeout=0.1)
                    
                    # Detecção de faces usando o modelo carregado
                    # Use a different variable name for dlib rectangles to avoid confusion
                    dlib_rects = self.face_detector(rgb_small_frame, 1) 
                    face_locations = [(rect.top(), rect.right(), rect.bottom(), rect.left()) 
                                    for rect in dlib_rects]
                    
                    face_names = []
                    recognized_known_user = None # Initialize here

                    # IMPORTANT: The local imports and the nested class FaceRecognitionLogin
                    # definition that follow here in your current file are highly problematic
                    # and should be refactored.
                    # For now, this initialization of recognized_known_user should cover the subsequent logic.
                    
                    # (Original local imports like import cv2, etc. were here - should be at top of file)
                    # (Original nested class FaceRecognitionLogin definition was here - should be removed/merged)

                    # Assuming the logic that was causing the error (original lines 223-253) follows:
                    if face_locations and self.known_face_encodings:
                        face_encodings_in_frame = []
                        # If you are using the liveness check logic, this part would be different.
                        # This is the older logic from your file:
                        for face_location_tuple in face_locations:
                            shape = self.shape_predictor(rgb_small_frame, 
                                                       dlib.rectangle(face_location_tuple[3], 
                                                                    face_location_tuple[0], 
                                                                    face_location_tuple[1], 
                                                                    face_location_tuple[2]))
                            current_face_encoding = np.array(self.face_recognition_model.compute_face_descriptor(
                                rgb_small_frame, shape))
                            face_encodings_in_frame.append(current_face_encoding)
                        
                        for current_face_encoding in face_encodings_in_frame:
                            face_distances = [np.linalg.norm(known_encoding - current_face_encoding) 
                                            for known_encoding in self.known_face_encodings]
                            
                            current_name = "Desconhecido" # Initialize current_name for each face
                            if face_distances: # Check if face_distances is not empty
                                best_match_index = np.argmin(face_distances)
                                if face_distances[best_match_index] < 0.6:
                                    current_name = self.known_face_names[best_match_index]
                                    if recognized_known_user is None: # Assign to recognized_known_user
                                        recognized_known_user = current_name
                            face_names.append(current_name)

                    if recognized_known_user:
                        self.is_logged_in = True
                        self.current_user = recognized_known_user
                    else:
                        self.is_logged_in = False
                        self.current_user = None
                    
                    # Scale face locations back for display if they are from a scaled frame
                    # Assuming 'scale=0.25' was used in process_frame_for_recognition, multiply by 4
                    scaled_face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                                            for (top, right, bottom, left) in face_locations]
                    
                    if not self.result_queue.full():
                        self.result_queue.put((scaled_face_locations, face_names))
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no worker: {e}")
                time.sleep(0.1) # Add a small delay to prevent tight loop on continuous errors
                continue

    def capture_face(self, name):
        """Captura e processa o rosto da pessoa - versão otimizada"""
        print(f"\nCadastrando novo usuário: {name}")
        print("Posicione seu rosto no centro da tela")
        print("Pressione 'q' para finalizar o cadastro")
        
        try:
            video_capture = self.setup_camera()
            captured_encodings = 0
            max_encodings = 5  # Reduzido para 5 encodings de alta qualidade
            frame_count = 0
            
            while captured_encodings < max_encodings:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Processa detecção a cada 3 frames
                if frame_count % 3 == 0:
                    try:
                        # Usa o detector de faces do dlib
                        face_locations = self.face_detector(rgb_frame, 1)
                        
                        if len(face_locations) == 1:  # Apenas processa se houver exatamente uma face
                            face_location = face_locations[0]
                            # Obtém landmarks faciais
                            shape = self.shape_predictor(rgb_frame, face_location)
                            # Calcula encoding facial usando o modelo ResNet
                            face_encoding = np.array(self.face_recognition_model.compute_face_descriptor(
                                rgb_frame, shape))
                            
                            # Salva o encoding
                            self.save_face_encoding(name, face_encoding)
                            captured_encodings += 1
                            print(f"Encoding {captured_encodings} capturado!")
                            
                            # Desenha retângulo na face
                            rect = face_location
                            cv2.rectangle(frame, (rect.left(), rect.top()), 
                                        (rect.right(), rect.bottom()), (0, 255, 0), 2)
                    
                    except Exception as e:
                        print(f"Erro ao processar frame: {e}")
                
                cv2.putText(frame, f"Encodings: {captured_encodings}/{max_encodings}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Cadastro Facial', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
        
        print(f"\nCadastro finalizado! {captured_encodings} encodings salvos.")
        # Recarrega os encodings após novo cadastro
        self.load_known_faces()

    def load_known_faces(self):
        """Carrega os encodings faciais do arquivo JSON"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        print("\nCarregando encodings faciais...")
        
        if not os.path.exists(ENCODINGS_FILE):
            print("Nenhum encoding facial encontrado.")
            return
        
        try:
            with open(ENCODINGS_FILE, 'r') as f:
                encodings = json.load(f)
            
            for name, face_encodings in encodings.items():
                for encoding in face_encodings:
                    self.known_face_encodings.append(np.array(encoding))
                    self.known_face_names.append(name)
                print(f"✓ Carregados {len(face_encodings)} encodings para: {name}")
                
        except Exception as e:
            print(f"Erro ao carregar encodings: {e}")

    def draw_face_box(self, frame, face_locations, face_names):
        """Desenha as caixas e nomes das faces detectadas"""
        for i, (top, right, bottom, left) in enumerate(face_locations):
            name = face_names[i] if i < len(face_names) else "Desconhecido"
            
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def run_login_system(self):
        """Executa o sistema de login com reconhecimento facial"""
        print("\nIniciando sistema de login facial...")
        print("Pressione 'q' para sair")
        print("Pressione 'c' para cadastrar novo usuário")
        
        # Inicia worker thread para processamento
        worker_thread = threading.Thread(target=self.face_recognition_worker, daemon=True)
        worker_thread.start()
        
        try:
            video_capture = self.setup_camera()
            frame_count = 0
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Envia frame para processamento apenas a cada 6 frames
                if frame_count % 6 == 0 and not self.frame_queue.full():
                    rgb_small_frame = self.process_frame_for_recognition(frame)
                    try:
                        self.frame_queue.put_nowait(rgb_small_frame)
                    except queue.Full:
                        pass
                
                # Obtém resultados do processamento
                try:
                    self.face_locations, self.face_names = self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Desenha as faces detectadas
                self.draw_face_box(frame, self.face_locations, self.face_names)
                
                # Exibe mensagem de boas-vindas ou cadastro
                if self.is_logged_in:
                    cv2.putText(frame, f"Bem-vindo(a) {self.current_user}!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Face nao reconhecida. Pressione 'c' para cadastrar", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Sistema de Login Facial', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    name = input("\nDigite o nome do novo usuário: ").strip()
                    if name:
                        self.capture_face(name)
                        self.is_logged_in = False
                        self.current_user = None
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

def main():
    print("Sistema de Login com Reconhecimento Facial")
    print("=" * 50)
    
    login_system = FaceRecognitionLogin()
    login_system.run_login_system()

if __name__ == "__main__":
    main()